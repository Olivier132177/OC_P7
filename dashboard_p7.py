import dash
import dash_table
import dash_bootstrap_components as dbc
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output, State
import plotly.express as px
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from joblib import dump,load
from plotly.subplots import make_subplots
from sklearn.preprocessing import Normalizer
import plotly.figure_factory as ff
import matplotlib.pyplot as plt


modele=load('modele_sauvegarde.joblib')

data_test_apres=pd.read_csv('df_test_prep_pour_dashboard.csv', index_col=0)
data_test_avant=pd.read_csv('df_test_avant_transfo_pour_dashboard.csv', index_col=0)
data_train=pd.read_csv('df_train_pour_dashboard.csv', index_col=0)
coefficients=pd.read_csv('coefficients.csv',index_col=0)
inte=round(modele.intercept_[0],3)
intercep=pd.Series([inte,np.abs(inte),1,inte],name='Valeur_d_origine')
#application_final=pd.read_csv('data/df_pour_dashboard.csv', index_col=0)
coefficients.sort_values('Coef').tail(20)
def boxplot(variable,dossier):

    fig4=go.Figure()
    moyenne=data_test_avant.loc[:,variable].mean()
    mediane=data_test_avant.loc[:,variable].median()
    mini=data_test_avant.loc[:,variable].min()
    maxi=data_test_avant.loc[:,variable].max()
    ecarty=data_test_avant.loc[:,variable].std()
    q1=data_test_avant.loc[:,variable].quantile(0.25)
    q3=data_test_avant.loc[:,variable].quantile(0.75)

    if moyenne-ecarty< mini:
        stm1=mini
    else:
        stm1=moyenne-ecarty

    if moyenne-ecarty*2< mini:
        stm2=mini
    else:
        stm2=moyenne-ecarty*2

    if moyenne+ecarty > maxi:
        stp1=maxi
    else:
        stp1=moyenne+ecarty

    if moyenne+ecarty*2 > maxi:
        stp2=maxi
    else:
        stp2=moyenne+ecarty*2

    fig4.add_trace(go.Indicator(  #Valeur client vs mediane
        mode='gauge+delta',#'gauge+number+delta'
        title='Valeur client vs mediane:',
        value=data_test_avant.loc[dossier,variable],
        delta={'reference':mediane},        
        gauge={
              'shape':'bullet',
              'axis': {'range': [mini, maxi]},
              'bar':{'color':'black'
                    },
            'threshold':{'value':mediane,
                         'thickness':1,
                         'line':{'width':3,
                                 'color':'blue'
                                }
                        },
            'steps':[
                    {'range': [mini, q1], 'color': 'cyan'},
                    {'range': [q1, mediane], 'color': 'darkturquoise'},
                    {'range': [mediane,q3], 'color': 'darkturquoise'},
                    {'range': [q3,maxi], 'color': 'cyan'}
                    ]
              },
        domain={'x':[0.2,0.9],'y':[0.65,1]}
        ))
    
    fig4.add_trace(go.Indicator( #Valeur client vs moyenne
        mode='gauge+delta',
        title='Valeur client vs moyenne:',
        value=data_test_avant.loc[dossier,variable],
        delta={'reference':moyenne},        
        gauge={
              'shape':'bullet',
              'axis': {'range': [mini, maxi]},
              'bar':{'color':'black'
                    },
            'threshold':{'value':moyenne,
                         'thickness':1,
                         'line':{'width':3,
                                 'color':'red'
                                }
                        },
            'steps':[
                    {'range': [mini, stm2], 'color': 'mistyrose'},
                    {'range': [stm2, stm1], 'color': 'lightcoral'},
                    {'range': [stm1,moyenne], 'color': 'brown'},
                    {'range': [moyenne,stp1], 'color': 'brown'},
                    {'range': [stp1,stp2], 'color': 'lightcoral'},
                    {'range': [stp2,maxi], 'color': 'mistyrose'}


                    ]
              },
        domain={'x':[0.2,0.9],'y':[0,0.35]}
        ))

    fig5=go.Figure()
    fig5.add_trace(go.Box(x=[data_test_avant.loc[dossier,variable]],name='Dossier client',marker_color='black'))
    fig5.add_trace(go.Box(x=data_train.loc[data_train['label']==0,variable],name='Dossiers sans incident', marker_color='green', boxmean='sd'))
    fig5.add_trace(go.Box(x=data_train.loc[data_train['label']==1,variable],name='Dossiers avec défaut de paiement',marker_color='red', boxmean='sd'))
    fig5.add_trace(go.Box(x=data_train.loc[:,variable],name='Ensemble des dossiers', marker_color='blue', boxmean='sd'))
    fig6=jauge(ind=False,prob=50,visib=False)
    
    fig4.update_layout(height=300)
    fig5.update_layout(height=450)
    fig6.update_layout(height=10)
    
    return fig4,fig5,fig6

def prediction(dossier):
    clas=modele.predict(data_test_apres.loc[dossier].to_numpy().reshape(1,-1))
    prob=modele.predict_proba(data_test_apres.loc[dossier].to_numpy().reshape(1,-1))[0][1]
    return clas, prob 

def retourne_val(dossier,variable):
    return data_test_avant.loc[dossier,variable]

def impact_coef(dossier,nombre,interce):
    ana_coef=coefficients.join(data_test_apres.loc[dossier])
    ana_coef['impact']=round(ana_coef['Coef']*ana_coef[dossier],3)
    interce.index=ana_coef.columns
    #ana_coef=ana_coef.append(interce)
    c_neg_1=ana_coef.loc[ana_coef['impact']<0,:]
    c_pos_1=ana_coef.loc[ana_coef['impact']>0,:]
    c_neg=c_neg_1.sort_values('impact').iloc[:nombre]
    c_pos=c_pos_1.sort_values('impact',ascending=False).iloc[:nombre]
    c_neg=c_neg.sort_values('impact',ascending=False)
    c_pos=c_pos.sort_values('impact')
    
    fig_neg = px.bar(c_neg, y=c_neg.index, x='impact', color_discrete_sequence=['lime'], orientation='h',text='impact', title='Top 10 élements favorables')
    fig_pos = px.bar(c_pos, y=c_pos.index, x='impact', color_discrete_sequence=['red'], orientation='h',text='impact',title='Top 10 élements défavorables')
    scor= ana_coef['impact'].sum()
    return ana_coef, fig_neg,fig_pos,scor



def jauge(ind,prob,visib):
    if ind:
        val=round(prob*100,1)
    else:
        val=50
    figu = go.Figure(go.Indicator(
        mode = "gauge+number",
        value = val,
        visible=visib,
        number = {'suffix': "%"},
        domain = {'x': [0, 1], 'y': [0, 1]},
        title = {'text': 'Probabilité de défaut de paiement'},
        gauge = {
            'axis': {'range': [0, 100]},
            'bar':{'color':'black'},
            'steps':[ {'range': [0, 33], 'color': 'green'},
                      {'range': [33, 50], 'color': 'yellow'},
                      {'range': [50, 67], 'color': 'yellow'},                     
                      {'range': [67, 100], 'color': 'red'}],
            #'shape':'bullet',
            'threshold': {
                'value':50,
                'line':{
                'color':'grey' }
                        }
                }                       
        )
        )
    #figu.update_layout(margin=dict(l=0, r=0, t=0, b=0))
    return figu

application_numbers=data_test_apres.index[:100]

fig_neg=px.bar()
fig_pos=px.bar()
score_tot=go.Figure()
histo=px.histogram()



def categ_plots(vari,vale):
    variable=vari
    df_gb1=data_train[variable].value_counts(normalize=True)
    df_gb2=data_train.groupby([variable,'label']).count().iloc[:,0].reset_index()
    df_gb2=pd.pivot_table(df_gb2,columns='label',index=variable)
    df_gb2=df_gb2.droplevel(0,axis=1).fillna(0)
    df_gb3=df_gb2.T
    nor=Normalizer(norm='l1')
    df_gb2t=pd.DataFrame(nor.fit_transform(df_gb2))
    df_gb3t=pd.DataFrame(nor.fit_transform(df_gb3))

    df_gb2t.index=df_gb2.index
    df_gb2t.columns=df_gb2.columns
    df_gb2t=df_gb2t.reset_index()
    df_gb2t=pd.melt(df_gb2t,id_vars=variable)
    df_gb2t['label'].map({0:'Dossiers sans incident',1:'Dossiers avec défaut de paiement'})
    df_gb2t['value']=round(df_gb2t['value']*100,1)
    df_gb3t
    df_gb3t.index=df_gb3.index
    df_gb3t.columns=df_gb3.columns
    df_gb3t=df_gb3t.rename(index={0:'Dossiers sans incident',1:'Dossiers avec défaut de paiement'})
    df_gb3t=df_gb3t.reset_index()
    df_gb3t=pd.melt(df_gb3t,id_vars='label')
    df_gb3t['value']=round(df_gb3t['value']*100,1)
    df_gb1=round(df_gb1*100,1)
    tab_coul=df_gb1.index==vale
    print(vale)
    print(df_gb1)
    coul_bar=['cyan' if i==True else 'grey' for i in tab_coul]
    print(coul_bar)
    
    graph1=px.bar(df_gb1,y=df_gb1.values,x=df_gb1.index, #orientation='h'
    color_discrete_sequence=coul_bar,color=df_gb1.index, text=df_gb1.values, 
    labels={'y':'Pourcentage (%)','index':'Catégories'})

    #graph1.update_layout(visible =tab_coul)

    test4=px.bar(df_gb2t,x=variable,y='value',color='label',barmode='stack',
    labels={'value':'Pourcentage (%)',variable:'Catégories'}, text='value')
    
    test5=px.bar(df_gb3t,y='label',x='value',color=variable, orientation='h',
    labels={'label':'','value':'Pourcentage (%)'}, text='value')
    graph1.update_layout(height=275)
    test4.update_layout(height=275)
    test5.update_layout(height=200)
    
    return graph1,test4,test5

fig1=jauge(False,prob=50,visib=False)
fig_score=go.Figure(go.Indicator(value=0))

app = dash.Dash(external_stylesheets=[dbc.themes.BOOTSTRAP],suppress_callback_exceptions=True)

navbar = dbc.NavbarSimple(
    children=[
        dbc.Button("Sidebar", outline=True, color="secondary", className="mr-1", id="btn_sidebar"),
    ],
    brand="Prêt à Dépenser : Dashboard",
    brand_href="#",
    color="dark",
    dark=True,
    fluid=True,
)


# the style arguments for the sidebar. We use position:fixed and a fixed width
SIDEBAR_STYLE = {
    "position": "fixed",
    "top": 62.5,
    "left": 0,
    "bottom": 0,
    "width": "16rem",
    "height": "100%",
    "z-index": 1,
    "overflow-x": "hidden",
    "transition": "all 0.5s",
    "padding": "0.5rem 1rem",
    "background-color": "#f8f9fa",
}

SIDEBAR_HIDEN = {
    "position": "fixed",
    "top": 62.5,
    "left": "-16rem",
    "bottom": 0,
    "width": "16rem",
    "height": "100%",
    "z-index": 1,
    "overflow-x": "hidden",
    "transition": "all 0.5s",
    "padding": "0rem 0rem",
    "background-color": "#f8f9fa",
}

# the styles for the main content position it to the right of the sidebar and
# add some padding.
CONTENT_STYLE = {
    "transition": "margin-left .5s",
    "margin-left": "18rem",
    "margin-right": "2rem",
    "padding": "2rem 1rem",
    "background-color": "#f8f9fa",
}

CONTENT_STYLE1 = {
    "transition": "margin-left .5s",
    "margin-left": "2rem",
    "margin-right": "2rem",
    "padding": "2rem 1rem",
    "background-color": "#f8f9fa",
}

page1=html.Div([
                html.Div('Sélectionner un numéro de dossier:'),
                dcc.Dropdown(id='liste_1',options=[{'label': i, 'value': i} for i in application_numbers]),
                html.H4(id='phrase',style={'height':50}),  
                html.Div([
                        html.Div([
                            dcc.Graph(id='jauge_target',figure=fig1),
                            dcc.Graph(id='score',figure=fig_score),
                                ],style={'display':'flex'}),
                        ],style={'width':1600,'height':200,'display':'flex'}),
                html.Div([
                        html.Div([   
                          
                            dcc.Graph(id='positif',figure=fig_pos)
                                 ],style={'width':700}),
                        html.Div([
                            
                            dcc.Graph(id='negatif',figure=fig_neg)
                                 ],style={'width':700}),
                        ],style={'display':'flex','width':1600,'height':600}),

               ],style={'width':1600,'height':800})

page2=html.Div([ 
               html.Div('Numéro de dossier:'),
               dcc.Dropdown(id='liste_2a',options=[{'label': i, 'value': i} for i in application_numbers]),
               html.Div('Sélectionner une variable:'),
               dcc.Dropdown(id='liste_2b',options=[{'label': i, 'value': i} for i in data_train.iloc[:,:-1].columns]),
               html.H4(id='lib_p2'),
               dcc.Graph(id='analyse_variable'),#,style={'height':300}),
               dcc.Graph(id='analyse_variable2'),
               dcc.Graph(id='analyse_variable3')
                          
                ])
page3=html.Div([
                html.H1('Elements du dossier'),
                html.Div('Sélectionner un numéro de dossier :'),
                dcc.Dropdown(id='liste_3a',options=[{'label': i, 'value': i} for i in application_numbers]),
                dcc.RadioItems(id='boutons',options=[
                                        {'label':"Informations client",'value':'info'},
                                        {'label':"Caractéristiques du prêt",'value':'carac'},
                                        {'label':'Elements du dossier','value':'elem'}
                                        ], value='info',inputStyle={"margin-left": "20px"}),
                html.Div(id='type_info',children='Informations client'),
                dash_table.DataTable(id='table',columns=[{'name':i,'id':i} for i in ['Variable','Valeur']],
                style_cell={'width':300}),
                
                ])

sidebar = html.Div(
    [
        html.H2("Menu", className="display-4"),
        html.Hr(),
        html.P(
            "Attribution Prêt Client", className="lead"
        ),
        dbc.Nav(
            [
                dbc.NavLink("Evaluation", href="/page-1", id="page-1-link", active=True),
                dbc.NavLink("Comparaison avec les autres dossiers", href="/page-2", id="page-2-link"),
                dbc.NavLink("Détails du dossier", href="/page-3", id="page-3-link"),

            ],
            vertical=True,
            pills=True,
        ),
    ],
    id="sidebar",
    style=SIDEBAR_STYLE,
)

content = html.Div(
    id="page-content",
    style=CONTENT_STYLE)

app.layout = html.Div(
    [
        dcc.Store(id='side_click'),
        dcc.Location(id="url"),
        navbar,
        sidebar,
        content,
    ],
)


@app.callback(
    [
        Output("sidebar", "style"),
        Output("page-content", "style"),
        Output("side_click", "data"),
    ],

    [Input("btn_sidebar", "n_clicks")],
    [
        State("side_click", "data"),
    ]
)
def toggle_sidebar(n, nclick):
    if n:
        if nclick == "SHOW":
            sidebar_style = SIDEBAR_HIDEN
            content_style = CONTENT_STYLE1
            cur_nclick = "HIDDEN"
        else:
            sidebar_style = SIDEBAR_STYLE
            content_style = CONTENT_STYLE
            cur_nclick = "SHOW"
    else:
        sidebar_style = SIDEBAR_STYLE
        content_style = CONTENT_STYLE
        cur_nclick = 'SHOW'

    return sidebar_style, content_style, cur_nclick

# this callback uses the current pathname to set the active state of the
# corresponding nav link to true, allowing users to tell see page they are on

@app.callback(
    [Output(f"page-{i}-link", "active") for i in range(1, 4)],
    [Input("url", "pathname")],
)
def toggle_active_links(pathname):
    if pathname == "/":
        # Treat page 1 as the homepage / index
        return True, False, False
    return [pathname == f"/page-{i}" for i in range(1, 4)]


@app.callback(Output("page-content", "children"), [Input("url", "pathname")])
def render_page_content(pathname):
    print(pathname)
    if pathname in ["/", "/page-1"]:
        return page1
    elif pathname == "/page-2":
        return page2
    elif pathname == "/page-3":
        return page3
    # If the user tries to reach a different page, return a 404 message
    return dbc.Jumbotron(
        [
            html.H1("404: Not found", className="text-danger"),
            html.Hr(),
            html.P(f"The pathname {pathname} was not recognised..."),
        ]
    )

@app.callback( #page1
    Output('jauge_target', 'figure'),
    Output('score','figure'),
    Output('negatif','figure'),
    Output('positif','figure'),
    Output('phrase','children'),
    Input('liste_1', 'value')
            )
def update_page1(ind):
    if ind:
        clas,prob = prediction(ind)
        fig1=jauge(ind,prob, visib=True)
        _,tneg,tpos,_=impact_coef(ind,nombre=10,interce=intercep)
        fig_score=go.Figure(go.Indicator(title={'text':'Score du dossier :'}\
            ,value=round(np.log(prob/(1-prob)),2)))
        if clas ==1:
            phra="Refus du dossier en raison d'une probabilité de défaut de paiement supérieure à 50%"
        else:
            phra="Dossier accepté : la probabilité de défaut de paiement est inférieure à 50%"
        fig1.update_layout(margin=dict(l=20, r=20, t=50, b=20))
        fig_score.update_layout(margin=dict(l=20, r=20, t=50, b=20))
        tneg.update_layout(height=550)
        tpos.update_layout(height=550)
        
    else:
        fig1=jauge(ind=False,prob=50,visib=False)
        tneg=jauge(ind=False,prob=50,visib=False)
        tpos=jauge(ind=False,prob=50,visib=False)
        fig_score=go.Figure(go.Indicator(visible=False))
        phra=""
        
    return fig1, fig_score,tneg,tpos, phra

@app.callback(# page2
    Output('analyse_variable','figure'),
    Output('analyse_variable2','figure'),
    Output('analyse_variable3','figure'),  
     
      
    Output('lib_p2','children'),
    Input('liste_2b','value'),
    Input('liste_2a','value')
            )
def update_page2(variable, dossier):
    if variable:
        if dossier :
            vale=retourne_val(dossier,variable)
            phra_p2='{} du dossier {} : {}'.format(variable,dossier,vale)
            if np.isin(variable,data_train[:-1].select_dtypes('number').columns):
                fig4,fig5,fig6=boxplot(variable,dossier)
                            
            else :
                fig4,fig5,fig6=categ_plots(variable,vale)
                
        else:
            fig4=jauge(ind=False,prob=50,visib=False)
            fig5=jauge(ind=False,prob=50,visib=False)
            fig6=jauge(ind=False,prob=50,visib=False)
            
            phra_p2=''
    else :
            fig4=jauge(ind=False,prob=50,visib=False)
            fig5=jauge(ind=False,prob=50,visib=False)
            fig6=jauge(ind=False,prob=50,visib=False)
            
            phra_p2=''
    return fig4,fig5,fig6,phra_p2


@app.callback(# page3
    Output('table','data'),
    Output('type_info','children'),
    Input('liste_3a','value'),
    Input('boutons','value')
            )
def update_page3(dossier, val_bouton):
    col_info_client=['SEXE', 'POSSEDE_UNE_VOITURE', 'REVENUS_CLIENTS',
       'ACCOMPAGNATEUR', 'TYPE_DE_REVENUS', 'NIVEAU_D_ETUDES',
       'STATUT FAMILIAL', 'LOGEMENT_ACTUEL',
       'AGE', 'EMPLOYE DEPUIS',
       'PROFESSION','TYPE_SOCIETE']
    col_info_pret=['TYPE_CONTRAT', 'MONTANT_CREDIT', 'MONTANT ANNUITE', 'PRIX_DU_BIEN',
       'CREDIT_TERM', 'DUREE_DU_CREDIT']
    col_info_elements=['A_FOURNI_LE_DOCUMENT_3', 'A_FOURNI_LE_DOCUMENT_6',
       'A_FOURNI_LE_DOCUMENT_7', 
       'A_FOURNI_LE_DOCUMENT_11', 'A_FOURNI_LE_DOCUMENT_13',
       'A_FOURNI_LE_DOCUMENT_14', 'A_FOURNI_LE_DOCUMENT_15',
       'A_FOURNI_LE_DOCUMENT_16', 'A_FOURNI_LE_DOCUMENT_17',
       'A_FOURNI_LE_DOCUMENT_18', 'A_FOURNI_LE_DOCUMENT_20',
       'A_FOURNI_LE_DOCUMENT_21']

    if dossier:

        if val_bouton=='info':
            sub_col=col_info_client
            txtinfo='Informations Clients'           
        elif val_bouton=='carac':
            sub_col=col_info_pret
            txtinfo='Caractéristiques Prêt'           
        elif val_bouton=='elem':
            sub_col=col_info_elements
            txtinfo='Elements du dossier'           
        dos=data_test_avant.loc[dossier,sub_col].reset_index()
        dos.columns=['Variable','Valeur']
        dos= dos.to_dict('records')
    else :
        dos=[]
        txtinfo=''

    return dos,txtinfo

if __name__ == "__main__":
    app.run_server(debug=True, port=8086)
