"""SafePath AI - interactive 3D evacuation demonstration (educational simulation)."""
import math
import numpy as np
import pandas as pd
import networkx as nx
import plotly.graph_objects as go
import streamlit as st
from sklearn.ensemble import RandomForestClassifier

st.set_page_config(page_title='SafePath AI | 3D Evacuation', layout='wide')
st.title('SafePath AI | Integrated Demo + Automatic Fire Simulation')
st.caption('AI-assisted detection • Risk-aware routing • Multi-person 3D animation • Schematics • Scenario reports')
st.warning('Educational simulation with synthetic sensor data. Not validated for real emergency decisions.')

ROOMS = list('ABCDEFGHI')
EXITS = ['Exit 1', 'Exit 2', 'Exit 3']
POS = {'A':(0,8),'B':(5,8),'C':(10,8),'D':(0,4),'E':(5,4),'F':(10,4),'G':(0,0),'H':(5,0),'I':(10,0),'Exit 1':(-4,4),'Exit 2':(14,4),'Exit 3':(5,12)}
CORRIDORS = [('A','B',6),('B','C',8),('D','E',5),('E','F',9),('G','H',7),('H','I',6),('A','D',10),('D','G',8),('B','E',6),('E','H',7),('C','F',9),('F','I',5),('D','Exit 1',4),('F','Exit 2',6),('B','Exit 3',5)]

@st.cache_resource
def fire_model():
    rng=np.random.default_rng(42)
    t=rng.uniform(15,115,3000)
    smoke=rng.uniform(0,100,3000)
    label=(((t>65)&(smoke>40))|(smoke>80)).astype(int)
    clf=RandomForestClassifier(n_estimators=80,max_depth=8,random_state=42)
    clf.fit(np.column_stack([t,smoke]),label)
    return clf

if 'readings' not in st.session_state:
    st.session_state.readings=pd.DataFrame({'Room':ROOMS,'Temperature_C':[25.]*9,'Smoke_index':[8.]*9})

st.subheader('1. Simulated fire detection — Random Forest')
st.markdown('**System schematic:** `Synthetic temperature + smoke sensors → Random Forest → Fire probabilities → Block unsafe rooms/corridors → Dijkstra routes → 2D/3D simulation`')
with st.expander('AI model assumptions and validation (read before presenting)'):
    st.write('This classifier learns labels made by a simple rule, not actual fire observations. It is NOT a validated real-world fire alarm. Probability values are demonstration scores, not calibrated physical probabilities. Routing operates on an illustrative graph; no real building has been surveyed.')
    st.code('Synthetic label = (temperature > 65°C AND smoke > 40) OR (smoke > 80)',language='text')

sensor_mode=st.radio('Sensor input mode', ['Automatic Fire Simulation','Manual Sensor Testing'],horizontal=True,
                     help='Automatic generates BOTH synthetic sensor readings from a simulated fire; Manual keeps both sensor fields independent.')
if sensor_mode=='Automatic Fire Simulation':
    st.info('Automatic mode simulates a developing fire. Temperature and smoke rise together because the simulation updates two independent virtual sensors. This is NOT a physical fire-spread model.')
    auto_left,auto_right=st.columns(2)
    with auto_left:
        ignition_rooms=st.multiselect('Simulated ignition room(s)',ROOMS,default=['D'],key='auto_ignition_rooms')
    with auto_right:
        elapsed=st.slider('Simulated fire development (%)',0,100,0,5,key='fire_development')
    st.caption('0% = normal room readings. Raise the development slider to simulate a fire growing over time. This updates both temperature and smoke automatically, then sends them into the AI detector.')
    # A repeatable *illustrative* progression; no actual heat/smoke physics implied.
    synthetic=[]
    for room in ROOMS:
        stage=elapsed/100 if room in ignition_rooms else 0.0
        synthetic.append({'Room':room,'Temperature_C':round(25+75*stage,1),
                          'Smoke_index':round(8+90*(stage**1.15),1)})
    readings=pd.DataFrame(synthetic)
    st.dataframe(readings,hide_index=True,use_container_width=True)
    if ignition_rooms and elapsed==0:
        st.caption('Fire simulation has not started yet. Move the development slider above 0%.')
else:
    st.info('Manual mode: edit temperature and smoke separately. High temperature alone does not automatically imply smoke.')
    readings=st.data_editor(st.session_state.readings,disabled=['Room'],hide_index=True,
                            num_rows='fixed',key='sensor_editor',use_container_width=True)

threshold=st.slider('Fire probability threshold',0.5,0.99,0.75,0.01)
probs=fire_model().predict_proba(readings[['Temperature_C','Smoke_index']].astype(float).to_numpy())[:,1]
fire_auto=[name for name,prob in zip(readings['Room'],probs) if prob>=threshold]
st.write('**AI-detected rooms:** '+(', '.join(fire_auto) if fire_auto else 'None'))
prob_table=pd.DataFrame({'Room':readings['Room'],'Temperature_C':readings['Temperature_C'],
                         'Smoke_index':readings['Smoke_index'],
                         'AI_fire_score':np.round(probs,3),
                         'Decision':['FIRE' if p>=threshold else 'NORMAL' for p in probs]})
with st.expander('See AI scores and decisions for every room'):
    st.dataframe(prob_table,hide_index=True,use_container_width=True)
st.caption('AI detection is based on simulated data and synthetic training labels. The FIRE decision changes the route graph automatically; it is not proof of real-world detection performance.')

st.subheader('2. Scenario controls')
c1,c2,c3=st.columns(3)
with c1:
    start=st.selectbox('Person location',ROOMS,index=4)
    additional_fire=st.multiselect('Manual room fires',ROOMS)
    extra_people=st.multiselect('Additional people (start rooms)',[r for r in ROOMS if r != start],default=['A','I'])
with c2:
    blocked_labels=st.multiselect('Burning / blocked corridors',[f'{a} — {b}' for a,b,_ in CORRIDORS])
    closed_exits=st.multiselect('Closed exits',EXITS)
with c3:
    congestion=st.slider('General corridor congestion (%)',0,100,10)
    frame=st.slider('Starting animation progress (%)',0,100,0)
    show_labels=st.checkbox('Show room labels',True)
fire_rooms=set(fire_auto)|set(additional_fire)
blocked=set(blocked_labels)

G=nx.Graph()
G.add_nodes_from(POS)
for a,b,d in CORRIDORS:
    G.add_edge(a,b,distance=float(d),weight=float(d)*(1+congestion/100))
safe=G.copy()
for room in fire_rooms:
    if room in safe:
        safe.remove_node(room)
for a,b,_ in CORRIDORS:
    if f'{a} — {b}' in blocked and safe.has_edge(a,b):
        safe.remove_edge(a,b)
for ex in closed_exits:
    if ex in safe:
        safe.remove_node(ex)

# Each burning starting room is temporarily reconnected ONLY for its own
# hypothetical escape attempt. All other burning rooms remain excluded.
# This does not certify that leaving the burning room is physically safe.
def candidate_routes(origin):
    routing_graph = safe.copy()
    if origin in fire_rooms:
        routing_graph.add_node(origin)
        for neighbor in G.neighbors(origin):
            # Never enter another burning room or a closed exit, and never
            # cross a manually blocked/burning corridor.
            if neighbor not in safe:
                continue
            if (f'{origin} — {neighbor}' in blocked or
                    f'{neighbor} — {origin}' in blocked):
                continue
            routing_graph.add_edge(origin, neighbor, **G[origin][neighbor])
    candidates = []
    for ex in EXITS:
        if ex not in routing_graph:
            continue
        try:
            path = nx.shortest_path(
                routing_graph, origin, ex, weight='weight', method='dijkstra'
            )
            candidates.append((nx.path_weight(routing_graph, path, 'weight'), ex, path))
        except (nx.NetworkXNoPath, nx.NodeNotFound):
            pass
    return sorted(candidates, key=lambda x: x[0])

routes = candidate_routes(start)
best_path = routes[0][2] if routes else None

st.subheader('3. Route result')
if best_path:
    distance=nx.path_weight(G,best_path,'distance')
    st.success('Route: '+' → '.join(best_path))
    st.write(f'Exit: **{routes[0][1]}** · Distance: **{distance:.1f} m** · Illustrative weighted cost: **{routes[0][0]:.1f}**')
else:
    st.error('No route available in this scenario.')
if start in fire_rooms:
    st.error('Starting room has fire: any displayed exit path is hypothetical; safe passage has NOT been verified. Follow professional emergency instructions in a real fire.')

# Calculate a separate dynamically updated exit route for every simulated occupant.
people_starts=[start]+[r for r in extra_people if r != start]
def find_route(origin):
    candidates = candidate_routes(origin)
    return candidates[0][2] if candidates else None

all_routes={person:find_route(person) for person in people_starts}
st.caption(f'People simulated: {len(people_starts)} · Routes found: {sum(route is not None for route in all_routes.values())}')

# Corridor schematic, detailed distances and scenario evidence in one place.
st.subheader('4. Building schematic and route evidence')
left,right=st.columns([3,2])
with left:
    import plotly.graph_objects as go
    pfig=go.Figure()
    for a,b,d in CORRIDORS:
        x1,y1=POS[a];x2,y2=POS[b]
        is_blocked=f'{a} — {b}' in blocked or a in fire_rooms or b in fire_rooms or a in closed_exits or b in closed_exits
        pfig.add_trace(go.Scatter(x=[x1,x2],y=[y1,y2],mode='lines',line=dict(color='#ef4444' if is_blocked else '#94a3b8',width=5 if is_blocked else 3,dash='dash' if is_blocked else 'solid'),hovertemplate=f'{a} ↔ {b} | {d} illustrative meters<extra></extra>',showlegend=False))
        pfig.add_annotation(x=(x1+x2)/2,y=(y1+y2)/2,text=str(d)+' m',showarrow=False,font=dict(size=10,color='#475569'),bgcolor='#ffffff')
    for idx,(origin,path) in enumerate(all_routes.items()):
        if path:
            pfig.add_trace(go.Scatter(x=[POS[n][0] for n in path],y=[POS[n][1] for n in path],mode='lines',line=dict(color=['#16a34a','#8b5cf6','#06b6d4','#ec4899'][idx%4],width=7),name=f'Person {idx+1}',hovertemplate='%{text}<extra></extra>',text=path))
    for name in POS:
        x,y=POS[name]
        is_exit=name in EXITS
        pfig.add_trace(go.Scatter(x=[x],y=[y],mode='markers+text',text=[name],textposition='top center',textfont=dict(size=14),marker=dict(size=19 if is_exit else 26,color='#ef4444' if name in fire_rooms else ('#dc2626' if name in closed_exits else '#16a34a' if is_exit else '#dbeafe'),line=dict(width=2,color='#334155')),showlegend=False,hovertemplate=f'{name}<extra></extra>'))
    pfig.update_layout(height=525,margin=dict(l=0,r=0,t=20,b=0),xaxis=dict(visible=False,range=[-6,17]),yaxis=dict(visible=False,range=[-3,15],scaleanchor='x',scaleratio=1),paper_bgcolor='#f8fafc',plot_bgcolor='#f8fafc',legend=dict(orientation='h'))
    st.plotly_chart(pfig,use_container_width=True)
    st.caption('2D floor-network schematic: line labels are illustrative edge distances, not measured building dimensions. Red dotted lines indicate blocked or unsafe connections.')
with right:
    st.markdown('**Exit decisions for all occupants**')
    table=[]
    for i,(origin,path) in enumerate(all_routes.items(),1):
        table.append({'Occupant':f'P{i}','Start':origin,'Exit':path[-1] if path else 'NONE','Path':' → '.join(path) if path else 'NO SAFE GRAPH ROUTE','Distance (illustrative m)':round(sum(G[u][v]['distance'] for u,v in zip(path,path[1:])),2) if path else None})
    results=pd.DataFrame(table)
    st.dataframe(results,hide_index=True,use_container_width=True)
    st.download_button('Download scenario results CSV',results.to_csv(index=False).encode('utf-8'),'safepath_scenario_results.csv','text/csv')
    st.markdown('**Corridor connection list (schematic)**')
    edge_table=pd.DataFrame([{'From':a,'To':b,'Distance (illustrative m)':d,'State':'BLOCKED' if f'{a} — {b}' in blocked or a in fire_rooms or b in fire_rooms or a in closed_exits or b in closed_exits else 'OPEN'} for a,b,d in CORRIDORS])
    st.dataframe(edge_table,hide_index=True,height=260,use_container_width=True)

with st.expander('Functional schematic — full data flow and components'):
    st.markdown('''
    1. **Input layer:** simulated temperature (°C) and smoke index (unitless) from each room.
    2. **AI layer:** synthetic-training Random Forest estimates room fire scores.
    3. **Decision layer:** probabilities above threshold and manually selected fires become blocked nodes; corridor/exit closures remove connections.
    4. **Graph layer:** NetworkX models A–I rooms, 15 connections, and 3 exits with illustrative edge lengths.
    5. **Routing layer:** Dijkstra calculates a viable minimum-weight path for each occupant separately.
    6. **Visualization layer:** 2D building-network diagram and 3D open-roof architecture show routes, people, fire, and exits.
    7. **Scenario evidence:** route table and CSV export enable demonstration and comparison.
    ''')
    st.info('No actual hardware/sensors are connected. This is a software-only, educational prototype.')

st.subheader('5. 3D building and multi-person evacuation animation')


fig = go.Figure()

# Architectural view: solid room panels, realistic door openings, tiled corridors.
def mesh_box(x0,x1,y0,y1,z0,z1,color,opacity=1):
    xx=[x0,x1,x1,x0,x0,x1,x1,x0]
    yy=[y0,y0,y1,y1,y0,y0,y1,y1]
    zz=[z0,z0,z0,z0,z1,z1,z1,z1]
    tris=[(0,2,1),(0,3,2),(0,1,5),(0,5,4),(1,2,6),(1,6,5),(2,3,7),(2,7,6),(3,0,4),(3,4,7),(4,5,6),(4,6,7)]
    fig.add_trace(go.Mesh3d(x=xx,y=yy,z=zz,i=[t[0] for t in tris],j=[t[1] for t in tris],k=[t[2] for t in tris],color=color,opacity=opacity,flatshading=True,hoverinfo='skip',showlegend=False))

def line_3d(points,color,width=6,dash=None):
    fig.add_trace(go.Scatter3d(x=[p[0] for p in points],y=[p[1] for p in points],z=[p[2] for p in points],mode='lines',line=dict(color=color,width=width),hoverinfo='skip',showlegend=False))

def sign(x,y,z,label,color='#1e293b',size=13):
    fig.add_trace(go.Scatter3d(x=[x],y=[y],z=[z],mode='text',text=[label],textfont=dict(color=color,size=size),hoverinfo='skip',showlegend=False))

# Ground and main corridor network. Distances are graph weights, not to-scale geometry.
mesh_box(-6.2,16.2,-3.0,15.0,-.26,-.20,'#dbe5ee')
for a,b,_ in CORRIDORS:
    x0,y0=POS[a]; x1,y1=POS[b]
    if abs(x1-x0)>abs(y1-y0):
        mesh_box(min(x0,x1),max(x0,x1),y0-.72,y0+.72,-.08,.02,'#bfcbd7')
        line_3d([(min(x0,x1),y0-.71,.035),(max(x0,x1),y0-.71,.035)],'#eef4f9',3)
        line_3d([(min(x0,x1),y0+.71,.035),(max(x0,x1),y0+.71,.035)],'#eef4f9',3)
    else:
        mesh_box(x0-.72,x0+.72,min(y0,y1),max(y0,y1),-.08,.02,'#bfcbd7')
        line_3d([(x0-.71,min(y0,y1),.035),(x0-.71,max(y0,y1),.035)],'#eef4f9',3)
        line_3d([(x0+.71,min(y0,y1),.035),(x0+.71,max(y0,y1),.035)],'#eef4f9',3)

# Nine individual open-roof rooms. Each opening faces an adjacent graph edge.
for room in ROOMS:
    x,y=POS[room]; r=1.47; ht=1.7; wall=.12; door=.57
    burning=room in fire_rooms
    floor='#fbd5cf' if burning else '#e8f4fd'
    panels='#df8c87' if burning else '#9baebe'
    mesh_box(x-r,x+r,y-r,y+r,.025,.10,floor)
    # subtle tile seams
    for dx in (-.5,.5):line_3d([(x+dx,y-r,.108),(x+dx,y+r,.108)],'#c9dbe7',2)
    for dy in (-.5,.5):line_3d([(x-r,y+dy,.109),(x+r,y+dy,.109)],'#c9dbe7',2)
    adjacent=set(G.neighbors(room))
    for side in ('N','S','E','W'):
        opening=any((POS[n][1]>y+.2 if side=='N' else POS[n][1]<y-.2 if side=='S' else POS[n][0]>x+.2 if side=='E' else POS[n][0]<x-.2) for n in adjacent)
        spans=[(-r,-door),(door,r)] if opening else [(-r,r)]
        if side in ('N','S'):
            yy=y+r if side=='N' else y-r
            for lo,hi in spans:
                mesh_box(x+lo,x+hi,yy-wall/2,yy+wall/2,.10,ht,panels)
                mesh_box(x+lo,x+hi,yy-wall/2,yy+wall/2,ht,ht+.07,'#536e84')
            if opening:
                # Door lintel and dark doorframe: entrance is visibly open below.
                mesh_box(x-door,x+door,yy-wall/2,yy+wall/2,1.40,ht,'#536e84')
                for dx in (-door,door):mesh_box(x+dx-.055,x+dx+.055,yy-.09,yy+.09,.10,1.40,'#536e84')
        else:
            xx=x+r if side=='E' else x-r
            for lo,hi in spans:
                mesh_box(xx-wall/2,xx+wall/2,y+lo,y+hi,.10,ht,panels)
                mesh_box(xx-wall/2,xx+wall/2,y+lo,y+hi,ht,ht+.07,'#536e84')
            if opening:
                mesh_box(xx-wall/2,xx+wall/2,y-door,y+door,1.40,ht,'#536e84')
                for dy in (-door,door):mesh_box(xx-.09,xx+.09,y+dy-.055,y+dy+.055,.10,1.40,'#536e84')
    # Room labeling at floor level (visible above furniture).
    if show_labels: sign(x,y+.72,.15,room,'#172b3b',17)
    # Small furnishing islands to make this read as an interior, not a graph of boxes.
    if not burning:
        mesh_box(x-.62,x-.08,y-.56,y-.18,.11,.24,'#b6c9d7')
        mesh_box(x-.59,x-.11,y-.53,y-.21,.24,.26,'#708ca2')
        mesh_box(x+.30,x+.65,y+.18,y+.57,.11,.32,'#d2b89a')
    else:
        # stylized fire cone and smoke plume, clearly indicating simulated hazard
        fig.add_trace(go.Cone(x=[x],y=[y],z=[.17],u=[0],v=[0],w=[1.0],anchor='tail',sizemode='absolute',sizeref=.85,colorscale=[[0,'#f59e0b'],[1,'#ef4444']],showscale=False,hoverinfo='skip',showlegend=False))
        fig.add_trace(go.Scatter3d(x=[x+.13,x+.24,x+.30],y=[y+.05,y+.10,y+.19],z=[1.08,1.48,1.93],mode='markers',marker=dict(size=[6,9,12],color='#64748b',opacity=.42),showlegend=False,hoverinfo='skip'))

# Exit platforms and signs.
for ex in EXITS:
    x,y=POS[ex]; closed=ex in closed_exits
    mesh_box(x-.75,x+.75,y-.72,y+.72,-.06,.07,'#fecaca' if closed else '#bbf7d0')
    mesh_box(x-.63,x+.63,y+.38,y+.50,.08,.92,'#be123c' if closed else '#15803d')
    sign(x,y,.98,('CLOSED' if closed else ex.upper()),'#be123c' if closed else '#166534',12)

for a,b,_ in CORRIDORS:
    if f'{a} — {b}' in blocked:
        xa,ya=POS[a];xb,yb=POS[b]
        line_3d([(xa,ya,.35),(xb,yb,.35)],'#e11d48',13)

# Three-dimensional markers and individually computed evacuation trails for multiple people.
person_palette=['#f59e0b','#8b5cf6','#06b6d4','#ec4899','#eab308','#2563eb','#f97316','#14b8a6','#a855f7']
def moving_location(path,percentage):
    if not path:
        return None
    if len(path)==1:
        return POS[path[0]]
    edges=list(zip(path[:-1],path[1:]))
    lengths=[G[a][b]['distance'] for a,b in edges]
    progress=percentage/100*sum(lengths)
    for (a,b),length in zip(edges,lengths):
        if progress<=length:
            t=progress/length if length else 0
            return (POS[a][0]+t*(POS[b][0]-POS[a][0]),POS[a][1]+t*(POS[b][1]-POS[a][1]))
        progress-=length
    return POS[path[-1]]

# Moving people must be Scatter3d traces, not static Mesh3d boxes.
# Plotly frames update each occupant's marker/text without rebuilding the architecture.
moving_traces=[]
active_people=[]
for person_idx,(origin,path) in enumerate(all_routes.items()):
    color=person_palette[person_idx%len(person_palette)]
    if path:
        vertices=[(*POS[n],.34+person_idx*.035) for n in path]
        line_3d(vertices,'#16a34a' if person_idx==0 else color,12 if person_idx==0 else 7)
        point=moving_location(path,frame)
    else:
        point=POS[origin]
    x,y=point
    trace_idx=len(fig.data)
    fig.add_trace(go.Scatter3d(
        x=[x],y=[y],z=[.82],
        mode='markers+text',
        marker=dict(size=13,color=color,symbol='circle',line=dict(color='#ffffff',width=2)),
        text=['P1' if person_idx==0 else f'P{person_idx+1}'],
        textposition='top center',textfont=dict(color='#152536',size=13),
        hovertemplate=f'Person {person_idx+1} — starting room {origin}<extra></extra>',
        showlegend=False
    ))
    moving_traces.append(trace_idx)
    active_people.append((origin,path,color))

# 51 frames animate the actual marker positions. The original Streamlit slider
# remains useful for manually choosing the initial scene position.
animation_frames=[]
for pct in range(0,101,2):
    animated=[]
    for person_idx,(origin,path,color) in enumerate(active_people):
        x,y=moving_location(path,pct) if path else POS[origin]
        animated.append(go.Scatter3d(
            x=[x],y=[y],z=[.82],
            mode='markers+text',
            marker=dict(size=13,color=color,symbol='circle',line=dict(color='#ffffff',width=2)),
            text=['P1' if person_idx==0 else f'P{person_idx+1}'],
            textposition='top center',textfont=dict(color='#152536',size=13)
        ))
    animation_frames.append(go.Frame(name=str(pct),data=animated,traces=moving_traces))
fig.frames=animation_frames

fig.update_layout(
    height=820,
    scene=dict(xaxis=dict(visible=False,range=[-7.5,17.5]),
               yaxis=dict(visible=False,range=[-5.0,17.0]),
               zaxis=dict(visible=False,range=[-.4,3.8]),
               aspectmode='manual',aspectratio=dict(x=1.13,y=1.0,z=.28),
               camera=dict(eye=dict(x=1.35,y=1.45,z=2.0),center=dict(x=0,y=0,z=-.18)),
               bgcolor='#f1f5f9'),
    margin=dict(l=0,r=0,t=80,b=0),paper_bgcolor='#f1f5f9',showlegend=False,
    updatemenus=[dict(type='buttons',showactive=False,x=0.02,y=1.1,xanchor='left',yanchor='top',
        buttons=[
            dict(label='▶ PLAY — move people',method='animate',
                 args=[None,dict(frame=dict(duration=160,redraw=True),transition=dict(duration=0),
                                 fromcurrent=False,mode='immediate')]),
            dict(label='❚❚ PAUSE',method='animate',
                 args=[[None],dict(frame=dict(duration=0,redraw=True),transition=dict(duration=0),mode='immediate')])
        ])])
st.markdown('Use the **▶ PLAY** control in the 3D graphic to animate the occupants along the calculated routes.')
st.plotly_chart(fig,use_container_width=True,key='safepath_moving_people')
st.caption('3D architecture is illustrative, not an as-built layout. Press ▶ PLAY above the figure; the slider sets the initial position before playback. Model probabilities and geometry are synthetic.')

st.subheader('6. Acceptance tests for presentation')
st.markdown('''
| Test | How to demonstrate | Expected behavior |
|---|---|---|
| Basic evacuation | Clear fire and closures; choose E | A valid path to an open exit |
| Dynamic rerouting | Set fire at D | Paths avoid D and update after widget change |
| Exit failure | Close Exit 3 | No proposed path ends at Exit 3 |
| Corridor closure | Block B — Exit 3 | That edge cannot appear in a returned route |
| Multi-person | Select two additional starting rooms | Different occupants get individual routes |
| No-route handling | Close all three exits | The app shows no viable graph route |
| Fire detection | Set room sensor temperature to 90 and smoke index to 90 | Simulated model flags the room at suitable threshold |
| Playback | Click PLAY in the 3D figure | Markers travel along their computed path |
''')
st.warning('Safety limitations: this is a demonstration, not a certified life-safety decision system. Graph distances, crowd model, fire spread, sensors and ML training are illustrative. Do not use for real evacuation guidance.')

st.info('Try: add fire in D, close Exit 1, add occupants in A and I, then press ▶ PLAY inside the 3D chart. Routes update whenever you change the scenario.')
