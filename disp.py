from dash.dependencies import Output, Input
import dash_core_components as dcc
import dash_html_components as html
from dash_html_components.Div import Div
import plotly.express as px
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import dash
import dash_table


application_final=pd.read_csv('data/df_pour_dashboard.csv', index_col=0)
application_final['AGE']=application_final['DAYS_BIRTH_x']//365.25
application_numbers=application_final.index[:50]
application_final.iloc[2]
df_minmaxmoy=pd.read_csv('data/df_minmaxmoy.csv',index_col=0)
for i in df_minmaxmoy.columns:
    df_minmaxmoy[i]=df_minmaxmoy[i].apply(lambda x : round(x,2))

selec=['AMT_INCOME_TOTAL', 'AMT_CREDIT', 'AMT_ANNUITY', 'AMT_GOODS_PRICE',
        'DAYS_EMPLOYED','DAYS_REGISTRATION', 'REGION_RATING_CLIENT',
       'REGION_RATING_CLIENT_W_CITY', 
        'CREDIT_INCOME_PERCENT', 'ANNUITY_INCOME_PERCENT','DAYS_EMPLOYED_PERCENT', 'CREDIT_TO_GOODS_RATIO',
       'INCOME_TO_EMPLOYED_RATIO', 'INCOME_TO_BIRTH_RATIO', 'CREDIT_LENGTH', 'AVERAGE_LOAN_TYPE',
       'ACTIVE_LOANS_PERCENTAGE',
       'AVG_ENDDATE_FUTURE', 'DEBT_CREDIT_RATIO', 'OVERDUE_DEBT_RATIO',
       'AVG_CREDITDAYS_PROLONGED']
df_minmaxmoy2=df_minmaxmoy.loc[selec]
df_minmaxmoy2=df_minmaxmoy2.reset_index().rename(columns={'index':'variables'})

col_table=['variables', 'client','moyenne','ecart_type','minimum','maximum','mediane']

app2 = dash.Dash(__name__)

def tableau_comp(inde):
    donnees_client=application_final.loc[inde]
    donnees_client=donnees_client[selec]
    nouv_tabl=df_minmaxmoy2.merge(donnees_client,left_on='variables',right_index=True)
    nouv_tabl=nouv_tabl.rename(columns={inde:'client'})
    nouv_tabl=nouv_tabl[['variables', 'client','moyenne','ecart_type','minimum','maximum','mediane']]
    nouv_tabl['client']=nouv_tabl['client'].apply(lambda x : round(x,2))
    for i in nouv_tabl.columns:
        nouv_tabl[i]=nouv_tabl[i].astype('str')
    return nouv_tabl.to_dict('records')

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
    val_min=round(application_final[variable].min()*100,1)
    val_max=round(application_final[variable].max()*100,1)
    #val_med=round(application_final[variable].median()*100,1)
    if ind:
        val=round(application_final.loc[ind,variable]*100,1)
    else:
        val=50
    figu = go.Figure(go.Indicator(
        mode = "gauge+number",
        value = val,
        number = {'suffix': "%"},
        domain = {'x': [0, 1], 'y': [0, 1]},
        title = {'text': titre},
        
        gauge = {
            'axis': {'range': [0, 100]},
            'bar':{'color':'black', 'thickness':0.3 },
            'steps':[ {'range': [0, 33], 'color': 'green'},
                      {'range': [33, 50], 'color': 'yellow'},
                      {'range': [50, 67], 'color': 'yellow'},                     
                      {'range': [67, 100], 'color': 'red'}],
            #'shape':'bullet',
            'threshold': {
                'value':50,
                'thickness':0.9,
                'line':{
                'color':'grey','width':3 }
                        }
                }                       
        )
        )
    figu.update_layout(margin=dict(l=0, r=0, t=50, b=0))
    return figu

fig1=jauge('y_pred',False,'Risque de défaut de paiement :')

app2.layout=html.Div([ #bloc principal
                        html.H1('Prêt à dépenser'),
                        html.Div([#1ere section avec liste déroulante et saisie du numéro de client
                            html.Div('Saisir un numéro de client:'),
                            dcc.Dropdown(id='liste-derou',
                                        options=[{'label': i, 'value': i} for i in application_numbers],
                                        placeholder='Saisir un numéro de dossier')
                                ],style={'background-color':'pink','display':'inline','height':100}),
                        html.Div([ #2ème section avec les informations sous format texte
                                html.Div([ #partie de gauche avec les textes
                                    html.Div(children='Détail de la demande de prêt ', style={'font-size':18, 'font-weight':'bold'}),
                                    html.Div(children='Type de prêt : ',id='type_pret'),
                                    html.Div(children='Montant du crédit : ',id='montant_pret'),
                                    html.Div(children='Montant des annuités : ',id='montant_annuites'),                                 
                                    html.Div(children='Montant du bien : ',id='montant_bien'),                                    
                                    html.Div(children='Durée du crédit : ',id='duree_credit'),                                  
                                    html.H4(children='Statut du prêt :', id='label', style={'font-size':16, 'font-weight':'bold'}), #y_pred
                                        ],style={'background-color':'lightgray','display':'flow', 'width':'30%','height':180}),
                                html.Div([#partie du centre avec les informations clients
                                    html.Div(children='Information du client', style={'font-size':18, 'font-weight':'bold'}),
                                    html.Div(children='Age :', id='age'), # DAYS_BIRTH
                                    html.Div(children='Sexe : ',id='sexe'),
                                    html.Div(children='Statut Familial :', id='statut_familial'), #NAME_FAMILY_STATUS
                                    html.Div(children='Profession :', id='profession'), #NAME_INCOME_TYPE
                                    html.Div(children='Revenus :', id='revenus'), #AMT_INCOME_TOTAL
                                ],style={'background-color':'lightgray','display':'flow', 'width':'30%','height':180}),
                                dcc.Graph(id='jauge_target',figure=fig1,style={'margin':10})
                                        ],style={'display':'flex'}),
                                                        
                        html.Div([ #3ème section, du bas
                            html.Div([
                                    html.H3("Donnees clients comparées à celles de l'ensemble des clients"),
                                    #dcc.Dropdown(id='liste-derou_gauche',
                                    #    options=[{'label': i, 'value': i} for i in application_numbers]),
                                    dash_table.DataTable(id='table',columns=[{"name": i, "id": i} for i in col_table])
                                    ],style={'background-color':'gray','width':'40%'}), #partie de gauche
                            html.Div([
                                    html.H3("Points positifs du dossier")
                                    #dcc.Dropdown(id='liste-derou_centre',
                                    #    options=[{'label': i, 'value': i} for i in application_numbers])
                                    ],style={'background-color':'lime','width':'30%'}), #partie centrale
                            html.Div([
                                    html.H3('Points négatifs du dossier')
                                    #dcc.Dropdown(id='liste-derou_droite',
                                    #    options=[{'label': i, 'value': i} for i in application_numbers])
                                    ],style={'background-color':'brown','width':'30%'}) #partie de droite
                                ],style={'background-color':'yellow','display':'flex','height':700}
                                )
                        ])




@app2.callback(
    Output('jauge_target', 'figure'),
    Output('type_pret','children'),
    Output('montant_pret','children'),
    Output('montant_annuites','children'),
    Output('montant_bien','children'), 
    Output('duree_credit','children'), 
    Output('label','children'), 
    Output('age','children'), 
    Output('sexe','children'), 
    Output('statut_familial','children'), 
    Output('profession','children'),
    Output('revenus','children'),
    Output('table','data'),
    Input('liste-derou', 'value')
            )
def update_output_div(ind):
    if ind:
        fig1=jauge('y_pred',ind,'Risque de défaut de paiement :')
        txt1=('Type de prêt : {}'.format(application_final.loc[ind,'NAME_CONTRACT_TYPE'])),
        txt2=('Montant du crédit : {}$'.format(application_final.loc[ind,'AMT_CREDIT'])),                               
        txt3=('Montant des annuités : {}$'.format(application_final.loc[ind,'AMT_ANNUITY'])),
        txt4=('Montant du bien : {}$'.format(application_final.loc[ind,'AMT_GOODS_PRICE'])),                                   
        txt5=('Durée du crédit : {} mois'.format(round(application_final.loc[ind,'CREDIT_LENGTH']*12))),
        if application_final.loc[ind,'y_pred']<0.5:
            statut_pret='Accepté'
        else:
            statut_pret='Refusé'            
        txt6='Statut du prêt : {}'.format(statut_pret)
        txt7='Age : {} ans'.format(application_final.loc[ind,'AGE'])
        txt8='Sexe : {}'.format(application_final.loc[ind,'CODE_GENDER'])
        txt9='Statut familial : {}'.format(application_final.loc[ind,'NAME_FAMILY_STATUS'])
        txt10='Profession : {}'.format(application_final.loc[ind,'OCCUPATION_TYPE'])
        txt11='Revenus : {}$ / an'.format(application_final.loc[ind,'AMT_INCOME_TOTAL'])
        tabl1=tableau_comp(ind)

    else:
        fig1=jauge('y_pred',ind=False,titre='Risque de défaut de paiement :')
        txt1=''
        txt2=''
        txt3=''
        txt4=''
        txt5=''
        txt6=''
        txt7=''
        txt8=''
        txt9=''
        txt10=''
        txt11=''
        tabl1=[]
    return fig1,txt1,txt2,txt3,txt4,txt5,txt6,txt7,txt8,txt9,txt10,txt11,tabl1

if __name__ == '__main__':
    app2.run_server(debug=True)