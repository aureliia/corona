# Ici nous allons importer tous les modules nécessaires à l'éxecution du code 
import datetime
import os
import yaml

import numpy as np
import pandas as pd

import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output

from scipy.integrate import odeint
import matplotlib.pyplot as plt

from scipy.integrate import solve_ivp
from scipy.optimize import minimize

# Lecture du fichier d'environnement que nous avons crée dans le notebook: visualisation_epidemie
ENV_FILE = '../env.yaml'
with open(ENV_FILE) as f:
    params = yaml.load(f, Loader=yaml.FullLoader)

# Initialisation des chemins vers les fichiers
ROOT_DIR = os.path.dirname(os.path.abspath(ENV_FILE))
DATA_FILE = os.path.join(ROOT_DIR,
                         params['directories']['processed'],
                         params['files']['all_data'])

# Lecture du fichier de données
epidemie_df = (pd.read_csv(DATA_FILE, parse_dates=['Last Update'])
               .assign(day=lambda _df: _df['Last Update'].dt.date)
               .drop_duplicates(subset=['Country/Region', 'Province/State', 'day'])
               [lambda df: df['day'] <= datetime.date(2020, 3, 10)]
              )

# On définit les variables countries et country_sir qui donneront la liste de tous les pays présents dans notre base de données
countries = [{'label': c, 'value': c} for c in sorted(epidemie_df['Country/Region'].unique())]
country_sir = [{'label': c, 'value': c} for c in sorted(epidemie_df['Country/Region'].unique())]

# Création du site/ plateforme que l'on nomme Corona Virus Explorer
app = dash.Dash('Corona Virus Explorer')
app.layout = html.Div([
    html.H1(['Corona Virus Explorer'], style={'textAlign': 'center'}),
    dcc.Tabs([
        dcc.Tab(label='Time', children=[    # Création d'un premier onglet time
            html.Div([
                dcc.Dropdown(
                    id='country',           # Aide à la création d'une barre de sélection de pays
                    options=countries
                )
            ]),
            html.Div([
                dcc.Dropdown(
                    id='country2',         # Aide à la création d'une deuxième barre de sélection de pays
                    options=countries
                )
            ]),
            html.Div([
                dcc.RadioItems(
                    id='variable',         # Aide à la création de trois boutons: Confirmed, Deaths, Recovered, dont un à selectioner.
                    options=[
                        {'label': 'Confirmed', 'value': 'Confirmed'},
                        {'label': 'Deaths', 'value': 'Deaths'},
                        {'label': 'Recovered', 'value': 'Recovered'}
                    ],
                    value='Confirmed',
                    labelStyle={'display': 'inline-block'}
                )
            ]),
            html.Div([
                dcc.Graph(id='graph1')    #Définit le nom de notre graphique sur l'onglet numéro 1
            ]),   
        ]),
        dcc.Tab(label='Map', children=[   # Création du second onglet nommé Map
            dcc.Graph(id='map1'),
            dcc.Slider(
                id='map_day',             # Titre du graph : la date de la carte ( qui va s'ajuster)
                min=0,
                max=(epidemie_df['day'].max() - epidemie_df['day'].min()).days,   #Va afficher le nombre de jour en bas de la map
                value=0,
                #marks={i:str(date) for i, date in enumerate(epidemie_df['day'].unique())}
                marks={i:str(i) for i, date in enumerate(epidemie_df['day'].unique())}
            )  
        ]),
        
     dcc.Tab(label='SIR MODEL', children=[   # Création du troisième onglet SIR MODEL
            html.Div([
                dcc.Dropdown(
                    id='country_sir',
                    options=countries_sir   # Renvoie à la liste de pays disponible dans notre base que l'on va pouvoir selectionner
                )
            ]),
            
            html.Div([
                dcc.Graph(id='graph2')     # Nom que l'on attribue au graphique de cet onglet
            ]),   
        ])
    ]),
])

@app.callback(                            # Grâce au callback on va appliquer des fonctions en utilisant nos input et cela va aller                                            dans nos output: Ici c'est le callback du premier onglet
    Output('graph1', 'figure'),
    [
        Input('country', 'value'),       # Définition des input qui vont devoir être utilisés et que l'on va devoir selectionné sur                                          l'onglet
        Input('country2', 'value'),
        Input('variable', 'value'),        
    ]
)
def update_graph(country, country2, variable):  #Création du graphique 
    print(country)
    if country is None:                         # Si on ne choisit pas de pays, on affiche la somme de toutes les variables
        graph_df = epidemie_df.groupby('day').agg({variable: 'sum'}).reset_index()
    else:
        graph_df = (epidemie_df[epidemie_df['Country/Region'] == country]   # selon le pays choisit on affiche le total de tous les pays
                    .groupby(['Country/Region', 'day'])
                    .agg({variable: 'sum'})
                    .reset_index()
                   )
    if country2 is not None:                                      # Si aucun pays selectionné on affiche la somme de la variable sélectionnée 
        graph2_df = (epidemie_df[epidemie_df['Country/Region'] == country2]
                     .groupby(['Country/Region', 'day'])
                     .agg({variable: 'sum'})
                     .reset_index()
                    )

        
    #data : [dict(...graph_df...)] + ([dict(...graph2_df)] if country2 is not None else [])
        
    return {
        'data': [
            dict(
                x=graph_df['day'],                 # Définition des graphiques dans chaque 'dict': les axes, le type et le titre
                y=graph_df[variable],
                type='line',
                name=country if country is not None else 'Total'
            )
        ] + ([
            dict(
                x=graph2_df['day'],
                y=graph2_df[variable],
                type='line',
                name=country2
            )            
        ] if country2 is not None else [])
    }

@app.callback(                             # Callback du deuxième onglet 
    Output('map1', 'figure'),              # Utilisation des output et input défini dans la première partie du code 
    [
        Input('map_day', 'value'),
    ]
)
def update_map(map_day):                     # dénition du contenu du graphique
    day = epidemie_df['day'].unique()[map_day]           # On souhaite avoir les jours apparents sur notre carte
    map_df = (epidemie_df[epidemie_df['day'] == day]
              .groupby(['Country/Region'])               # Et que ce soit regroupé par pays
              .agg({'Confirmed': 'sum', 'Latitude': 'mean', 'Longitude': 'mean'})
              .reset_index()
             )
    print(map_day)
    print(day)
    print(map_df.head())
    return {                           # Définition du type de graph: création de la carte
        'data': [
            dict(
                type='scattergeo',
                lon=map_df['Longitude'],
                lat=map_df['Latitude'],
                text=map_df.apply(lambda r: r['Country/Region'] + ' (' + str(r['Confirmed']) + ')', axis=1),
                mode='markers',
                marker=dict(
                    size=np.maximum(map_df['Confirmed'] / 1_000, 5)
                )
            )
        ],
        'layout': dict(
            title=str(day),
            geo=dict(showland=True),
        )
    }

# 1 ère méthode testée pour le modèle SIR:

def get_country(self, country_sir):             # On appelle la première fonction pour définir une nouvelle variable country_sir dans notre base de données
    return (epidemie_df[epidemie_df['Country/Region'] == country_sir]
            .groupby(['Country/Region', 'day'])
            .agg({'Confirmed': 'sum', 'Deaths': 'sum', 'Recovered': 'sum'})
            .reset_index()
           )

def sumsq_error(parameters):           #Définition de la fonction qui va retourner la somme des carrés des erreurs de nos parametres beta et gamma
    
    beta, gamma = parameters

#msol = minimize(sumsq_error, [0.001, 0.1])         # on minimise cette somme 
#msol.x
# Création/définition de nos variables qui vont nous servir à calculer la solution optimale
total_population = 51_470_000
#infected_population = epidemie_df['Country/Region'].unique()['infected']
#nb_steps = len(infected_population)

    
def SIR(t, y):            #Définition/ création de notre modèle SIR (Succeptible , Infected, recovered)
    S = y[0]
    I = y[1]
    R = y[2]
    return([-beta*S*I, beta*S*I-gamma*I, gamma*I])     # Calcul des parametres optimaux de chaque Y, on va différencier chacun d'eux
    
    # Calcul de la solution optimale
    solution = solve_ivp(SIR, [0, nb_steps-1], [total_population, 1, 0], t_eval=np.arange(0, nb_steps, 1))  
    
    #Calcul de la propagation de l'épidémie grâce à la solution optimale et la population infectée
    return(sum((solution.y[1]-infected_population)**2))  

# Définition de notre fonction qui va nous servir à réaliser notre graphique avec les 3 courbes associées aux trois variables du modèle: Succeptible, infected et recovered
def plot_epidemia(solution, infected, susceptible=False):          
    fig = plt.figure(figsize=(12, 5))
    if susceptible:
        plt.plot(solution.t, solution.y[0])
    plt.plot(solution.t, solution.y[1])
 
    plt.plot(solution.t, solution.y[2])
    plt.plot(infected.reset_index(drop=True).index, infected, "k*:")
    if susceptible:
        plt.legend(["Susceptible", "Infected", "Recovered", "Original Data"])
    else:
        plt.legend(["Infected", "Recovered", "Original Data"])
    #plt.show()


# on va appeler nos fonctions et input définies plus haut dans notre code pour réaliser le graph 2 du troisème onglet
@app.callback(     
    Output('graph2', 'figure'),
    [
        Input('country_sir', 'value'),
        #Input('variable_sir', 'value'),        
    ]
)

#On va créer notre graphique grâce aux résultats trouvés précédemment
def update_graph(country_sir, variable_sir):        
    #days = epidemie_df['days'].unique()[variable_sir]
    #total_population = 51_470_000
    #infected_population = epidemie_df['Country/Region'].unique()['infected']
    #nb_steps = len(infected_population)
    #solution = solve_ivp(SIR, [0, nb_steps-1], [total_population, 1, 0], t_eval=np.arange(0, nb_steps, 1))
    plot_epidemia_df= plot_epidemia.epidemie_df['Country/Region'].unique()
    #S0 = 51_470_000
    #beta = 5.67e-3
    #gamma = 24.7
    #print (infected_population)
    #print(solution)
    #print(days)
    print(country_sir)
    print(plot_epidemia_df)
   
    if country is None:
        graph_df = epidemie_df.groupby('day').agg({variable: 'sum'}).reset_index()
    else:
        graph_df = (epidemie_df[epidemie_df['Country/Region'] == country_sir]
            .groupby(['Country/Region', 'day'])
            .agg({'Confirmed': 'sum', 'Deaths': 'sum', 'Recovered': 'sum' })
            .reset.index()
                  )
        
        return {
        'data': [
            dict(
                x=graph_df['day'],
                y=graph_df[variable],
                type='line',
                name=country if country is not None else 'Total'
                #S I R=-beta*S*I, beta*S*I-gamma*I, gamma*I
                #solution= sum((solution.y[1]-infected_population)**2))
                )
        ] 
    } 
                

# 2 ème méthode du model SIR

#class SIR:
    #def __init__(self,N,I0,S0,beta,gamma,days):     # Défintions et intialisation de nos variables initiales
       # """
        #N - Total population 
        #I0 - Initial number of infected individuals, 
        #S0 - Everyone else is susceptible to infection initially
        #Contact rate, beta, and mean recovery rate, gamma, (in 1/days)
        #"""
        
        #N= 51_470_000
        #S0 = 51_470_000
        #beta = 5.67e-3
        #gamma = 24.7
        #days = epidemie_df['days'].unique()[variable_sir]
        
        #epidemie_df.N = epidemie_df['totale_population']
        #epidemie_df.I0 = epidemie_df['infected_population'] 
        #epidemie_df.S0 = epidemie_df['totale_population']NETFLIX
        
        #epidemie_df.beta = beta
        #epidemie_df.gamma = gamma
        #epidemie_df.S0 = epidemie_df['totale_population'] - epidemie_df['infected_population'] 
        #epidemie_df.days= epidemie_df['day']

 # Création du modèle SIR avec la différenciation des paramètres de chaque variables du modèle (3 variables)   
    #@staticmethod
    #def _deriv(y, t, N, beta, gamma):
        #"""
        #The SIR model differential equations
        #"""
        #S, I, R = y
        #print(t)
        #dSdt = -beta[int(t)] * S * I / N
        #dIdt = beta[int(t)] * S * I / N - gamma * I
        #dRdt = gamma * I
        #return dSdt, dIdt, dRdt
    
    #def run(self):         # On se sert de nos paramètres optimaux calculés plus haut pour trouver la solution otpimale du modèle
        #Initial conditions vector
        #y0 = epidemie_df.S0, epidemie_df.I0, 0
        #Integrating the SIR equations over the time grid, t
        #t = list(range(0, epidemie_df.days))
        # Getting results
        #result = odeint(epidemie._deriv, y0, t, args=(epidemie.N, epidemie.beta, epidemie.gamma))
        #S, I, R = result.T
        #return S, I, R
    
    #@staticmethod                #On va afficher nos résultats sur un graphique à 3 courbes pour chacune des variables du modèle
    #def plot_results(S, I, R):
        #fig = plt.figure(facecolor='w')
        #ax = fig.add_subplot(111, axisbelow=True)
        #ax.plot(S/1000, 'b', alpha=0.5, lw=2, label='Susceptible')
        #ax.plot(I/1000, 'r', alpha=0.5, lw=2, label='Infected')
        #ax.plot(R/1000, 'g', alpha=0.5, lw=2, label='Recovered')
        #ax.set_xlabel('Time /days')
        #ax.set_ylabel('Number (1000s)')
                    
        #ax.set_ylim(0,1.2)
                    
        #ax.yaxis.set_tick_params(length=0)       # Défini les paramètres du graph: axes, types de lignes
        #ax.xaxis.set_tick_params(length=0)
        #ax.grid(b=True, which='major', c='w', lw=2, ls='-')
        #legend = ax.legend()
        #legend.get_frame().set_alpha(0.5)
        #for spine in ('top', 'right', 'bottom', 'left'):
            #ax.spines[spine].set_visible(False)
        #plt.show()
        
    #data : [dict(...graph_df...)] + ([dict(...graph2_df)] if country2 is not None else [])
        
    
 

if __name__ == '__main__':
    app.run_server(debug=True)
    