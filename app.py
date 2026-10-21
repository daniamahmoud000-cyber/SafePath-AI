"""SafePath AI competition research simulator. NOT for real emergency use."""
import math
import time
import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import streamlit as st
import plotly.graph_objects as go
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import accuracy_score, mean_absolute_error
from sklearn.model_selection import train_test_split

ROOMS = list('ABCDEFGHI')
EXITS = ['Exit 1', 'Exit 2', 'Exit 3']
POS = {'A':(1,3),'B':(2,3),'C':(3,3),'D':(1,2),'E':(2,2),'F':(3,2),
       'G':(1,1),'H':(2,1),'I':(3,1),'Exit 1':(0,2),'Exit 2':(4,2),'Exit 3':(2,4)}
# DEMONSTRATION distances only, NOT measured building distances
CORRIDORS = [('A','B',6),('B','C',8),('D','E',5),('E','F',9),('G','H',7),
             ('H','I',6),('A','D',10),('D','G',8),('B','E',6),('E','H',7),
             ('C','F',9),('F','I',5),('D','Exit 1',4),('F','Exit 2',6),('B','Exit 3',5)]

@st.cache_resource
def train_models():
    rng = np.random.default_rng(42)
    n = 4500
    temp = rng.uniform(18, 110, n)
    smoke = rng.uniform(0, 100, n)
    # Synthetic labels. Represents simulated behavior, NOT real fire ground truth.
    fire_score = .075*(temp-35) + .105*(smoke-30) + rng.normal(0,2.2,n)
    y_fire = (fire_score > 4.5).astype(int)
    Xf = np.column_stack([temp, smoke])
    xf_train,xf_test,yf_train,yf_test=train_test_split(Xf,y_fire,test_size=.25,random_state=42,stratify=y_fire)
    fire_model=RandomForestClassifier(n_estimators=90,max_depth=9,random_state=42)
    fire_model.fit(xf_train,yf_train)
    fire_accuracy=accuracy_score(yf_test,fire_model.predict(xf_test))

    people=rng.integers(0,90,n)
    width=rng.uniform(1.0,4.5,n)
    distance=rng.uniform(3,25,n)
    # Simulated slowdown target: minutes per corridor
    simulated_time=distance/(70/60) * (1 + .028*people/width) / 60
    simulated_time+=rng.normal(0,.04,n)
    simulated_time=np.maximum(.01,simulated_time)
    Xc=np.column_stack([people,width,distance])
    xc_train,xc_test,yc_train,yc_test=train_test_split(Xc,simulated_time,test_size=.25,random_state=42)
    crowd_model=RandomForestRegressor(n_estimators=90,max_depth=12,random_state=42,n_jobs=-1)
    crowd_model.fit(xc_train,yc_train)
    crowd_mae=mean_absolute_error(yc_test,crowd_model.predict(xc_test))
    return fire_model,crowd_model,fire_accuracy,crowd_mae

def graph_from_corridors(rows):
    g=nx.Graph()
    g.add_nodes_from(POS)
    for a,b,d in rows:
        if d<=0: raise ValueError('Corridor length must be positive')
        g.add_edge(a,b,distance=float(d))
    return g

def detect_sensor_fires(model, sensors, threshold):
    values=sensors[['Temperature_C','Smoke_index']].to_numpy(dtype=float)
    probs=model.predict_proba(values)[:,1]
    probs_by_room=dict(zip(sensors['Room'],probs))
    return {r for r,p in probs_by_room.items() if p>=threshold},probs_by_room

def route_plan(g, start, room_fires, corridor_fires, closed_exits, spread_steps,
               people, widths, crowd_model, risk_weight, crowd_weight, policy):
    safe=g.copy()
    # Smoke/fire spread is illustrative graph-hop expansion, NOT physical fire propagation.
    affected=set(room_fires)
    for _ in range(spread_steps):
        affected |= {n for r in list(affected) if r in g for n in g.neighbors(r) if n in ROOMS}
    # Keep the starting room even if it is burning, so a hypothetical
    # exit route can begin there. Other burning rooms remain blocked.
    # NOTE: the model cannot establish that exits from a burning room are safe.
    for r in room_fires:
        if r != start and r in safe:
            safe.remove_node(r)
    for a,b in corridor_fires:
        if safe.has_edge(a,b): safe.remove_edge(a,b)
    for ex in closed_exits:
        if ex in safe: safe.remove_node(ex)
    # risk expands around fires; smoke zone is penalized, not asserted safe
    for a,b,data in safe.edges(data=True):
        length=data['distance']
        if (a in affected or b in affected):
            risk=3 if (a in room_fires or b in room_fires) else 2
        elif any(n in affected for v in (a,b) for n in g.neighbors(v)):
            risk=1
        else:
            risk=0
        key=frozenset((a,b))
        occupancy=people.get(key,0)
        width=widths.get(key,2.0)
        minutes=float(crowd_model.predict(np.array([[occupancy,width,length]]))[0])
        minutes=max(.01,minutes)
        # Cost measured in equivalent minutes. Baseline time and risks use chosen multipliers.
        if policy=='Distance only': cost=length
        else: cost=minutes*(1+crowd_weight*occupancy/max(width,0.5)/20)+risk_weight*risk
        data.update(weight=cost,risk=risk,estimated_minutes=minutes)
    options=[]
    for ex in EXITS:
        if start not in safe or ex not in safe: continue
        try:
            path=nx.shortest_path(safe,start,ex,weight='weight',method='dijkstra')
            edges=list(zip(path,path[1:]))
            cost=nx.path_weight(safe,path,'weight')
            distance=sum(safe[a][b]['distance'] for a,b in edges)
            minutes=sum(safe[a][b]['estimated_minutes'] for a,b in edges)
            risk=sum(safe[a][b]['risk'] for a,b in edges)
            options.append(dict(exit=ex,path=path,cost=cost,distance=distance,minutes=minutes,risk=risk))
        except (nx.NetworkXNoPath,nx.NodeNotFound): pass
    best=min(options,key=lambda x:x['cost']) if options else None
    return safe,affected,best,options

def draw_graph(g, route, room_fires, affected, corridor_fires, closed_exits, start, person_at=None):
    fig,ax=plt.subplots(figsize=(11,6.5))
    nx.draw_networkx_edges(g,POS,ax=ax,width=2,edge_color='#aab5c0')
    colors=[]
    for n in g.nodes:
        if n in room_fires: colors.append('#d83c46')
        elif n in closed_exits: colors.append('#555d68')
        elif n==start: colors.append('#ffa124')
        elif n in affected: colors.append('#f4c65a')
        elif n in EXITS: colors.append('#65c98b')
        else: colors.append('#88c8ed')
    nx.draw_networkx_nodes(g,POS,ax=ax,node_color=colors,node_size=1550)
    nx.draw_networkx_labels(g,POS,ax=ax,font_size=9,font_weight='bold')
    if route:
        nx.draw_networkx_edges(g,POS,ax=ax,edgelist=list(zip(route['path'],route['path'][1:])),width=5,edge_color='#188d51')
    if corridor_fires:
        nx.draw_networkx_edges(g,POS,ax=ax,edgelist=list(corridor_fires),width=5,edge_color='#d83c46',style='dashed')
    nx.draw_networkx_edge_labels(g,POS,ax=ax,edge_labels={(a,b):f'{d["distance"]:g}m' for a,b,d in g.edges(data=True)},font_size=7)
    # Draw person marker on top of nodes and routes, even inside a burning room.
    if person_at in POS:
        x, y = POS[person_at]
        ax.scatter([x], [y], s=220, c='#20242c', edgecolors='white',
                   linewidths=2, zorder=12)
        ax.text(x, y + .29, 'PERSON', ha='center', color='#20242c',
                fontsize=9, weight='bold', zorder=13)
    ax.set_title('SafePath AI — Research Evacuation Simulation')
    ax.axis('off')
    fig.tight_layout()
    return fig


def draw_3d_building(g, route, room_fires, affected, corridor_fires, closed_exits, person_at):
    """Illustrative 3D schematic: not a surveyed architectural floor plan."""
    fig = go.Figure()
    # Slightly raised floor: schematic size units, not architectural dimensions.
    fig.add_trace(go.Mesh3d(
        x=[-0.55,4.55,4.55,-0.55], y=[0.40,0.40,4.55,4.55],
        z=[-0.05]*4, i=[0,0], j=[1,2], k=[2,3],
        color='#eaf0f5', opacity=0.34, hoverinfo='skip', showscale=False,
        name='Floor'))

    # Each room drawn as an open-topped, transparent wall box.
    def wall_box(name, x, y, color):
        w, h, z = 0.55, 0.55, 0.55
        xmin,xmax=x-w/2,x+w/2
        ymin,ymax=y-h/2,y+h/2
        xx=[xmin,xmax,xmax,xmin,xmin,xmax,xmax,xmin]
        yy=[ymin,ymin,ymax,ymax,ymin,ymin,ymax,ymax]
        zz=[0,0,0,0,z,z,z,z]
        # Four rectangular vertical faces, no roof.
        fig.add_trace(go.Mesh3d(x=xx,y=yy,z=zz,
            i=[0,0,1,1,2,2,3,3],j=[1,5,2,6,3,7,0,4],
            k=[5,4,6,5,7,6,4,7],color=color,
            opacity=0.19,hoverinfo='skip',showscale=False,name=name))
        for aa,bb in [(0,1),(1,2),(2,3),(3,0),(4,5),(5,6),(6,7),(7,4),(0,4),(1,5),(2,6),(3,7)]:
            fig.add_trace(go.Scatter3d(x=[xx[aa],xx[bb]],y=[yy[aa],yy[bb]],
                z=[zz[aa],zz[bb]],mode='lines',line=dict(color=color,width=3),
                hoverinfo='skip',showlegend=False))

    for n in ROOMS:
        x,y=POS[n]
        color=('#dc3545' if n in room_fires else '#e6ac37' if n in affected else '#468bc4')
        wall_box(n,x,y,color)

    # Corridor walkways and distance hover labels.
    for a,b,data in g.edges(data=True):
        x1,y1=POS[a]; x2,y2=POS[b]
        burning=frozenset((a,b)) in {frozenset(t) for t in corridor_fires}
        fig.add_trace(go.Scatter3d(
            x=[x1,x2],y=[y1,y2],z=[0.045,0.045],mode='lines',
            line=dict(color='#dc3545' if burning else '#8b99a9',width=12 if burning else 7),
            name='Blocked corridor' if burning else 'Corridor',showlegend=False,
            hovertemplate=f'{a} – {b}: {data["distance"]:g} m'+ '<extra></extra>'))

    for ex in EXITS:
        x,y=POS[ex]
        fig.add_trace(go.Scatter3d(x=[x],y=[y],z=[0.09],mode='markers+text',
            marker=dict(size=10,color='#6c757d' if ex in closed_exits else '#22a35c'),
            text=[ex],textposition='top center',name=ex,showlegend=False))

    if route:
        pts=[POS[n] for n in route['path']]
        fig.add_trace(go.Scatter3d(x=[p[0] for p in pts],y=[p[1] for p in pts],
            z=[0.14]*len(pts),mode='lines+markers',
            line=dict(color='#05ad58',width=13),marker=dict(size=5,color='#05ad58'),
            name='Calculated route'))

    for r in room_fires:
        x,y=POS[r]
        fig.add_trace(go.Scatter3d(x=[x],y=[y],z=[0.76],mode='markers+text',
            marker=dict(size=12,color='#f03b35',symbol='diamond'),
            text=['FIRE'],textposition='top center',name='Fire',showlegend=False))

    if person_at in POS:
        x,y=POS[person_at]
        fig.add_trace(go.Scatter3d(x=[x],y=[y],z=[0.78],mode='markers+text',
            marker=dict(size=13,color='#212938',symbol='circle',line=dict(color='white',width=3)),
            text=['PERSON'],textposition='top center',name='Person',showlegend=False))

    fig.update_layout(
        title='3D Research Building — Room Boxes & Evacuation Route',
        scene=dict(xaxis=dict(visible=False),yaxis=dict(visible=False),zaxis=dict(visible=False),
                   aspectmode='manual',aspectratio=dict(x=1.3,y=1.05,z=.30),
                   camera=dict(eye=dict(x=1.5,y=-1.7,z=1.35))),
        height=630,margin=dict(l=0,r=0,t=54,b=0),
        legend=dict(orientation='h',y=1.04),
        paper_bgcolor='rgba(0,0,0,0)'
    )
    return fig

st.set_page_config(page_title='SafePath AI | Competition Prototype',layout='wide')
st.title('SafePath AI — Emergency Evacuation Research Simulator')
st.warning('Educational research prototype ONLY. Do not use for real evacuation decisions. Detection and congestion models are trained on synthetic data.')
fire_model,crowd_model,fire_accuracy,crowd_mae=train_models()
base=graph_from_corridors(CORRIDORS)

with st.sidebar:
    st.header('Scenario controls')
    start=st.selectbox('Person location',ROOMS,index=4)
    mode=st.radio('Fire input',['Manual','AI on simulated sensor readings'])
    manual_rooms=st.multiselect('Rooms on fire (manual)',ROOMS,default=['F'] if mode=='Manual' else [])
    idx_labels=[f'{a} – {b}' for a,b,d in CORRIDORS]
    burn_corridor_labels=st.multiselect('Burning / impassable corridors',idx_labels)
    closed_exits=st.multiselect('Closed exits',EXITS)
    spread_steps=st.slider('Illustrative smoke-zone spread (graph steps)',0,2,0)
    policy=st.radio('Routing method',['Risk-aware','Distance only'])
    risk_weight=st.slider('Risk penalty (equivalent minutes / level)',0.0,5.0,1.5,.1)
    crowd_weight=st.slider('Congestion preference strength',0.0,3.0,1.0,.1)

st.subheader('1. Automatic fire recognition from simulated sensor readings')
st.caption('These numbers are simulated test inputs, not readings from physical sensors or cameras. The classifier is ML, but its synthetic accuracy does not validate real fire detection.')
if 'sensor_table' not in st.session_state:
    st.session_state.sensor_table=pd.DataFrame({'Room':ROOMS,'Temperature_C':[25.0]*9,'Smoke_index':[8.0]*9})
sensors=st.data_editor(st.session_state.sensor_table,disabled=['Room'],hide_index=True,use_container_width=True,num_rows='fixed',key='sensor_editor')
fire_threshold=st.slider('AI fire probability threshold',0.50,0.99,0.75,.01)
try:
    sensors['Temperature_C']=pd.to_numeric(sensors['Temperature_C'],errors='raise')
    sensors['Smoke_index']=pd.to_numeric(sensors['Smoke_index'],errors='raise')
    if sensors[['Temperature_C','Smoke_index']].isna().any().any(): raise ValueError('Missing sensor values')
    predicted_fires,probs=detect_sensor_fires(fire_model,sensors,fire_threshold)
except (ValueError,TypeError) as e:
    st.error(f'Invalid sensor input: {e}')
    st.stop()
room_fires=set(manual_rooms) if mode=='Manual' else predicted_fires
st.write('Detected/specified fire rooms:',', '.join(sorted(room_fires)) or 'None')
st.caption(f'Synthetic hold-out fire classification accuracy: {fire_accuracy:.1%}; synthetic congestion time MAE: {crowd_mae:.3f} min. NOT real-world performance.')

st.subheader('2. Corridor crowd conditions')
st.caption('Edit corridor crowd size and width. All measurements remain hypothetical until verified against an actual building floor plan.')
if 'crowd_table' not in st.session_state:
    st.session_state.crowd_table=pd.DataFrame([{'Corridor':f'{a} – {b}','People':8,'Width_m':2.0,'Distance_m':d} for a,b,d in CORRIDORS])
crowd_data=st.data_editor(st.session_state.crowd_table,disabled=['Corridor'],hide_index=True,use_container_width=True,num_rows='fixed',key='crowd_editor')
try:
    if len(crowd_data)!=len(CORRIDORS): raise ValueError('Corridor count changed')
    lengths=[];people={};widths={}
    for (a,b,_),(_,row) in zip(CORRIDORS,crowd_data.iterrows()):
        p=float(row['People']); w=float(row['Width_m']); d=float(row['Distance_m'])
        if not all(map(math.isfinite,(p,w,d))) or p<0 or w<=0 or d<=0: raise ValueError('Distances and widths must be positive; people must be nonnegative')
        lengths.append((a,b,d))
        people[frozenset((a,b))]=p
        widths[frozenset((a,b))]=w
except (ValueError,TypeError,KeyError) as e:
    st.error(f'Invalid corridor data: {e}')
    st.stop()
G=graph_from_corridors(lengths)
burning_corridors={tuple(CORRIDORS[idx_labels.index(lbl)][:2]) for lbl in burn_corridor_labels}

st.subheader('3. Route calculation and live rerouting')
safe,affected,best,options=route_plan(G,start,room_fires,burning_corridors,set(closed_exits),spread_steps,people,widths,crowd_model,risk_weight,crowd_weight,policy)
if start in room_fires:
    st.error('Starting room is on fire. A hypothetical route may begin there if a connected corridor is not blocked. This is NOT a verified safe escape or real emergency guidance.')
if not best:
    st.error('NO AVAILABLE ROUTE IN THIS SIMULATED GRAPH. Do not interpret this as an instruction to remain in place.')
else:
    st.success('Selected route: '+' → '.join(best['path']))
    c1,c2,c3=st.columns(3)
    c1.metric('Route distance',f'{best["distance"]:.1f} m')
    c2.metric('Estimated travel time',f'{best["minutes"]:.2f} min')
    c3.metric('Route risk score',f'{best["risk"]}')
    st.caption('Time and risk are model-based simulation indicators, not reliable emergency time or safety estimates.')
# Step-by-step movement and visible automatic playback.
auto_play = False
if best:
    route_signature = tuple(best['path'])
    scenario_signature = (route_signature, start, tuple(sorted(room_fires)),
                          tuple(sorted(map(tuple, burning_corridors))),
                          tuple(sorted(closed_exits)), spread_steps,
                          policy, risk_weight, crowd_weight,
                          tuple((round(float(row['People']), 4), round(float(row['Width_m']), 4),
                                 round(float(row['Distance_m']), 4)) for _, row in crowd_data.iterrows()))
    if st.session_state.get('animation_scenario') != scenario_signature:
        st.session_state.animation_scenario = scenario_signature
        st.session_state.animation_step = 0
    st.markdown('### Person movement demonstration')
    c_move, c_play, c_reset = st.columns(3)
    with c_move:
        if st.button('➡ Move one step', disabled=st.session_state.animation_step >= len(route_signature)-1):
            st.session_state.animation_step += 1
    with c_play:
        auto_play = st.button('▶ Play entire route', disabled=st.session_state.animation_step >= len(route_signature)-1)
    with c_reset:
        if st.button('↺ Restart movement'):
            st.session_state.animation_step = 0
    current_room = route_signature[st.session_state.animation_step]
    loc_text = st.empty()
    loc_text.info(f'PERSON NOW AT: {current_room} | Step {st.session_state.animation_step} / {len(route_signature)-1}')
    if start in room_fires:
        st.error('Fire at starting room: the following is only a hypothetical movement demonstration. Never interpret this as verified safe escape guidance.')
else:
    current_room = start
    loc_text = st.empty()
    st.warning('No movement possible: no available simulated route.')
view_2d, view_3d = st.tabs(['2D floor graph', '3D building visualization'])
with view_2d:
    slot_2d = st.empty()
with view_3d:
    slot_3d = st.empty()
    st.caption('3D schematic only: not real architectural geometry. Person location follows the same movement steps as 2D.')

def show_person(room):
    fig = draw_graph(G, best, room_fires, affected, burning_corridors,
                     set(closed_exits), start, person_at=room)
    slot_2d.pyplot(fig)
    plt.close(fig)
    slot_3d.plotly_chart(draw_3d_building(G, best, room_fires, affected,
                         burning_corridors, set(closed_exits), room),
                         use_container_width=True, key=f'building_{room}_{st.session_state.get("animation_step",0)}')

show_person(current_room)
if auto_play and best:
    # Animation occurs during this Streamlit run, no stale browser-tab state.
    for i in range(st.session_state.animation_step + 1, len(route_signature)):
        time.sleep(0.8)
        st.session_state.animation_step = i
        location = route_signature[i]
        loc_text.info(f'PERSON NOW AT: {location} | Step {i} / {len(route_signature)-1}')
        show_person(location)
    loc_text.success(f'PERSON REACHED {route_signature[-1]} in the simulation.')
if options:
    st.dataframe(pd.DataFrame([{'Exit':x['exit'],'Path':' → '.join(x['path']),'Distance (m)':round(x['distance'],1),'Sim. time (min)':round(x['minutes'],2),'Risk score':x['risk'],'Optimization cost':round(x['cost'],2)} for x in options]),hide_index=True,use_container_width=True)

st.subheader('4. Comparison: distance-only vs risk-aware')
comparison=[]
for label in ('Distance only','Risk-aware'):
    _,_,r,_=route_plan(G,start,room_fires,burning_corridors,set(closed_exits),spread_steps,people,widths,crowd_model,risk_weight,crowd_weight,label)
    if r: comparison.append({'Method':label,'Chosen exit':r['exit'],'Route':' → '.join(r['path']),'Distance (m)':round(r['distance'],1),'Simulated time (min)':round(r['minutes'],2),'Risk points':r['risk']})
if comparison:
    st.dataframe(pd.DataFrame(comparison),hide_index=True,use_container_width=True)
else:
    st.info('No route under either method for the current scenario.')
st.info('Dynamic rerouting: change a fire, crowd count, corridor, or exit above. Streamlit recalculates the simulation automatically.')
