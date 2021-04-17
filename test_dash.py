from dash.dependencies import Output, Input
import dash_core_components as dcc
import dash_html_components as html
import plotly.express as px
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import dash

application_final['CREDIT_TERM']
application_final=pd.read_csv('data/df_pour_dashboard.csv', index_col=0)
application_final['AGE']=application_final['DAYS_BIRTH_x']//365.25
application_numbers=application_final.index[:50]

application_final['AMT_INCOME_TOTAL']

def camembert(variable,ind, titre):
    donnees=application_final[variable].value_counts() 
    labels=donnees.index
    values=donnees.values
    if ind:
        cat=application_final.loc[ind,variable]
    else:
        cat=''
    fig = go.Figure(data=[go.Pie(labels=labels, values=values, textinfo='label+percent',
                             insidetextorientation='radial'
                            )])
    fig.update_layout(title_text=' {}: {}'.format(titre,cat))
    fig.update_layout(paper_bgcolor="LightSteelBlue")
    return fig

############
def jauge(variable, ind,titre):
    val_min=application_final[variable].min()
    val_max=application_final[variable].max()
    val_med=application_final[variable].median()
    if ind:
        val=application_final.loc[ind,'AGE']
    else:
        val=val_med
    figu = go.Figure(go.Indicator(
        mode = "gauge+number",
        value = val,
        domain = {'x': [0, 1], 'y': [0, 1]},
        title = {'text': titre},
        
        gauge = {
            'axis': {'range': [val_min, val_max]},
            'bar':{'color':'blue'},
            'shape':'bullet',
            'threshold': {
                'value':val_med,
                'thickness':0.9,
                'line':{
                'color':'orange','width':5 }
                        }
                }                       
        ),
        )
    figu.update_layout(paper_bgcolor="LightSteelBlue")
 
    return figu

fig2=jauge(variable='AGE',ind=False, titre='AGE')
fig5=jauge('AMT_INCOME_TOTAL',ind=False, titre='REVENUS')
fig4=camembert(variable='NAME_CONTRACT_TYPE',ind=False, titre='TYPE DE CONTRAT')

app = dash.Dash(__name__)

app.layout = html.Div(children=[
                html.Div(children='Numéro de client : '),
    dcc.Dropdown(
                id='liste-derou',
                options=[{'label': i, 'value': i} for i in application_numbers],
                placeholder='Saisir un numéro de dossier'),
    

    html.Div(children='Sexe : ',id='code-gender'),
    html.Div(children='Voiture :', id='voiture'), #FLAG_OWN_CAR
    #html.Div(children='Propriétaire du logement principal :', id='proprietaire'), #FLAG_OWN_REALTY
    #html.Div(children='Nombre d\'enfants :', id='enfants'), # CNT_CHILDREN
    html.Div(children='Age:', id='age'), # DAYS_BIRTH
    html.Div(children='Type de revenu :', id='type_revenus'), #NAME_INCOME_TYPE
    html.Div(children='Revenus :', id='revenus'), #AMT_INCOME_TOTAL
    html.Div(children='Statut Familial :', id='statut_familial'), #NAME_FAMILY_STATUS
    html.Div(children='Accord du prêt :', id='label'), #y_pred
    dcc.Graph(id='graph1',figure=fig2,style={'height': 200}),
    dcc.Graph(id='graph5',figure=fig5,style={'height': 200, 'padding':50}),#,style={'height': 200,'width':800}),
    dcc.Graph(id='graph4',figure=fig4)
        ],style={'width':'35%','display':'block','box-sizing': 'border-box','background-color':'pink'})


















@app.callback(
    Output('graph1', 'figure'),
    Output('graph5','figure'),
    Output('graph4', 'figure'),
    Output('code-gender', 'children'),
    Output('voiture', 'children'),
    Output('age', 'children'),
    Output('type_revenus', 'children'),
    Output('revenus', 'children'),
    Output('statut_familial', 'children'),
    Output('label', 'children'),
    Input('liste-derou', 'value'))
def update_output_div(ind):
    if ind:
        fig2=jauge('AGE',ind,'AGE')
        fig4=camembert('NAME_CONTRACT_TYPE',ind, 'TYPE DE CONTRAT')    
        fig5=jauge('AMT_INCOME_TOTAL',ind, titre='REVENUS')
        txt1='Sexe : {}'.format(application_final.loc[ind,['CODE_GENDER']][0])
        txt2='Voiture : {}'.format(application_final.loc[ind,['FLAG_OWN_CAR']][0])
        txt5='Age: {}'.format(application_final.loc[ind,['DAYS_BIRTH_x']][0]//365.25)
        txt6='Type de revenus : {}'.format(application_final.loc[ind,['NAME_INCOME_TYPE']][0])
        txt7='Revenus: {}'.format(application_final.loc[ind,['AMT_INCOME_TOTAL']][0])
        txt8='Statut Familial : {}'.format(application_final.loc[ind,['NAME_FAMILY_STATUS']][0])
        txt9='Le risque de défaut de paiement associé à ce prêt est de {}%'.format(round(application_final.loc[ind,['y_pred']][0]*100),1)
    else:
        fig2=jauge('AGE',ind=False,titre='AGE')
        fig4=camembert('NAME_CONTRACT_TYPE',ind=False, titre='TYPE DE CONTRAT')     
        txt1=''
        txt2=''
        txt5='' 
        txt6='' 
        txt7='' 
        txt8='' 
        txt9=''
        fig5=jauge('AMT_INCOME_TOTAL',ind=False, titre='REVENUS')
    return fig2,fig5,fig4, txt1,txt2,txt5,txt6,txt7,txt8,txt9

if __name__ == '__main__':
    app.run_server(debug=True)


