import os
import feelpp.core as fppc
from feelpp.toolboxes.cfpdes import *
has_scimba=True
try:
    from tools.scimba_pinns import Run_Poisson2D, Poisson_2D, PoissonDisk2D
    from scimba.equations import domain
except:
    has_scimba=False

import pyvista as pv
import pandas as pd
import numpy as np

import plotly.express as px
from plotly.subplots import make_subplots
import itertools
import torch
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap 
import numpy as np
import pyvista as pv

class ParametricDiffusion: 
    """
    Solves the parametric diffusion problem
    - div (a(x, y; mu) * grad (u)) = f  in Omega
      u                             = g  in boundary

    - with f, g, and a depending on the parameter mu
    """
    def __init__(self, dim=2, order=1):
        self.dim   = dim
        self.model = dict()
        self.order = order
        self.mu_value = None # Ajout pour stocker la valeur actuelle de mu

    def genGeometry(self, filename, h, shape = 'Rectangle'):
        """
        Generate a cube geometry following the dimension  self.dim
        """
        geo="""SetFactory("OpenCASCADE");
        h={};
        dim={};
        """.format(h, self.dim)
        if self.dim==2 :
            if shape == 'Rectangle':
                geo+="""
                Rectangle(1) = {0, 0, 0, 1, 1, 0};
                Characteristic Length{ PointsOf{ Surface{1}; } } = h;
                Physical Curve("Gamma_D") = {1,2,3,4};
                Physical Surface("Omega") = {1};
                """
            elif shape == 'Disk':
                geo += """
                Disk(1) = {0, 0, 0, 1.0};
                Characteristic Length{ PointsOf{ Surface{1}; } } = h;
                Physical Curve("Gamma_D") = {1};
                Physical Surface("Omega") = {1};
                """

        elif self.dim == 3:
            if shape == 'Box':
                geo += """
                Box(1) = {0, 0, 0, 1, 1, 1};
                Characteristic Length{ PointsOf{ Volume{1}; } } = h;
                Physical Surface("Gamma_D") = {1,2,3,4,5,6};
                Physical Volume("Omega") = {1};
                """

            elif shape == 'Sphere':
                geo += """
                Sphere(1) = {0, 0, 0, 1.0};
                Characteristic Length{ PointsOf{ volume{1}; } } = h;
                Physical Surface("Gamma_D") = {1};
                Physical Volume("Omega") = {1};
                """
        with open(filename, 'w') as f:
            f.write(geo)

    def getMesh(self, filename, h, shape = 'Rectangle',verbose=False):
        """create mesh"""
        import os
        for ext in [".msh",".geo"]:
            f=os.path.splitext(filename)[0]+ext
            if os.path.exists(f):
                os.remove(f)
        if verbose:
            print(f"generate mesh {filename} with h={h} and dimension={self.dim}")
        self.genGeometry(filename=filename,h=h, shape=shape)
        mesh = fppc.load(fppc.mesh(dim=self.dim,realdim=self.dim), filename, h)
        return mesh

    def feel_solver(self, filename, json, h, shape = 'Rectangle',verbose=False):
        if verbose:
            print(f"Solving the parametric diffusion problem for mu = {self.mu_value}, h = {h}...") # Ajout de l'affichage de mu
        diffusion_pb = cfpdes(dim=self.dim, keyword=f"cfpdes-{self.dim}d-p{self.order}") # Nom de la variable modifié
        diffusion_pb.setMesh(self.getMesh(f"omega-{self.dim}.geo", h=h, shape =shape, verbose=verbose))
        diffusion_pb.setModelProperties(json)
        diffusion_pb.init(buildModelAlgebraicFactory=True)
        diffusion_pb.printAndSaveInfo()
        diffusion_pb.solve()
        diffusion_pb.exportResults()
        measures = diffusion_pb.postProcessMeasures().values()
        return measures

    def scimba_solver(self, h, shape='Rectangle', dim = 2, verbose=False):
        if has_scimba:
            return
        if verbose:
            print(f"Solving a Poisson problem for h = {h}...")

        diff = self.diff.replace('{', '(').replace('}', ')')

        if shape == 'Disk':
            xdomain = domain.SpaceDomain(2, domain.DiskBasedDomain(2, center=[0.0, 0.0], radius=1.0))

        elif shape == 'Rectangle':
            xdomain = domain.SpaceDomain(2, domain.SquareDomain(2, [[0.0, 1.0], [0.0, 1.0]]))

        pde = Poisson_2D(xdomain, rhs=self.rhs, diff=diff, g=self.g, u_exact=self.u_exact) # Garde Poisson_2D car la structure de SciMBA est utilisée
        u , pinn = Run_Poisson2D(pde, epoch=1000) # Garde Run_Poisson2D

        return u

    def __call__(self,
                 h=0.05,                                       # mesh size
                 order=1,                                     # polynomial order
                 name='u',                                    # name of the variable u
                 rhs='0',                                     # right hand side (can depend on x, y, mu) # Modifié: valeur par défaut
                 mu=1.0,                                      # Parameter mu # Ajout du paramètre mu
                 diff='{1,0,0,1}',                            # diffusion coefficient (can depend on x, y, mu) # Modifié: valeur par défaut
                 g='0',                                       # Dirichlet boundary conditions (can depend on x, y, mu) # Modifié: valeur par défaut
                 gN='0',                                      # Neumann boundary conditions
                 shape='Rectangle',                           # domain shape (Rectangle, Disk)
                 geofile=None,                                # geometry file
                 plot=None,                                   # plot the solution
                 solver='feelpp',                             # solver
                 u_exact='0',                                 # exact solution (can depend on x, y, mu) # Modifié: valeur par défaut
                 grad_u_exact = '{0,0}'                       # gradient of exact solution # Modifié: valeur par défaut
                 ):
        """
        Solves the parametric diffusion problem where :
        - h is the mesh size
        - order the polynomial order
        - rhs is the expression of the right-hand side f(x,y, mu)
        - mu is the parameter for the diffusion coefficient a(x, y; mu)
        """
        a = 0.0 # Absorption coefficient
        self.h = h
        self.measures = dict()
        self.rhs = rhs
        self.g = g
        self.gN = gN
        self.u_exact = u_exact
        self.diff = diff
        self.mu_value = mu # Stockage de la valeur de mu

        # Définition du coefficient de diffusion dépendant de mu # MODIFICATION IMPORTANTE
        diffusion_coefficient = self.diff.format(mu=mu) # Permet d'utiliser {mu} dans la chaîne diff

        self.pb = cfpdes(dim=self.dim, keyword=f"cfpdes-{self.dim}d-p{self.order}")
        self.model = lambda order,dim=2,name="u": {
            "Name": "ParametricDiffusion", # Nom du modèle modifié
            "ShortName": "ParamDiff",      # Nom court modifié
            "Models":
            {
                f"cfpdes-{self.dim}d-p{self.order}":
                {
                    "equations":"parametric_diffusion" # Nom de l'équation modifié
                },
                "parametric_diffusion":{ # Définition de l'équation de diffusion paramétrique
                    "setup":{
                        "unknown":{
                            "basis":f"Pch{order}",
                            "name":f"{name}",
                            "symbol":"u"
                        },
                        "coefficients":{
                            "c": f"{diffusion_coefficient}:x:y" if self.dim == 2 else f"{diffusion_coefficient}:x:y:z", # Coefficient de diffusion dépendant de mu
                            "f": f"{rhs}:x:y:mu" if self.dim == 2 else f"{rhs}:x:y:z:mu",       # Terme source dépendant potentiellement de mu
                            "a": f"{a}"
                        }
                    }
                }
            },
            "Materials":
            {
                "Omega":
                {
                    "markers":["Omega"]
                }
            },
            "BoundaryConditions":
            {
                "parametric_diffusion": # Nom des conditions aux limites modifié
                {
                    "Dirichlet":
                    {
                        "g":
                        {
                            "markers":["Gamma_D"],
                            "expr":f"{g}:x:y:mu" if self.dim == 2 else f"{g}:x:y:z:mu" # Conditions Dirichlet dépendantes potentiellement de mu
                        }
                    },
                    "Neumann":
                    {
                        "gN":
                        {
                            "markers":["Gamma_D"],
                            "expr":f"{gN}:x:y:nx:ny" if self.dim == 2 else f"{gN}:x:y:z:nx:ny:nz"
                        }
                    }
                }
            },
            "PostProcess":
            {
                f"cfpdes-{self.dim}d-p{self.order}":
                {
                    "Exports":
                    {
                        "fields":["all"],
                        "expr":{
                            "rhs": f"{rhs}:x:y:mu" if self.dim == 2 else f"{rhs}:x:y:z:mu",
                            "u_exact" : f"{u_exact}:x:y:mu" if self.dim==2 else f"{u_exact}:x:y:z:mu",
                            "grad_u_exact" : f"{grad_u_exact}:x:y:mu" if self.dim==2 else f"{grad_u_exact}:x:y:z:mu"
                        }
                    },
                    "Measures" :
                    {
                        "Norm" :
                        {
                            "parametric_diffusion" : # Nom de la mesure modifié
                            {
                               "type":["L2-error", "H1-error"],
                               "field":f"parametric_diffusion.{name}", # Nom du champ modifié
                               "solution": f"{u_exact}:x:y:mu" if self.dim==2 else f"{u_exact}:x:y:z:mu", # Solution exacte dépendant de mu
                               "grad_solution": f"{grad_u_exact}:x:y:mu" if self.dim==2 else f"{grad_u_exact}:x:y:z:mu", # Gradient de la solution exacte dépendant de mu
                               "markers":"Omega",
                               "quad":6
                            }
                        },
                        "Statistics":
                        {
                            "mystatA":
                            {
                                 "type":["min","max","mean","integrate"],
                                 "field":f"parametric_diffusion.{name}" # Nom du champ modifié
                            }
                        }
                    }
                }
            }
        }

        fn = None
        if geofile is None:
            fn = f'omega-{self.dim}.geo'
            self.genGeometry(fn, self.h, shape=shape)
        else:
            fn = geofile

        # Solving
        parametric_diffusion_json = self.model
        self.measures = self.feel_solver(filename=fn, h=self.h, shape =shape, json=parametric_diffusion_json(order=self.order,dim=self.dim), verbose=True)

        # Plots (conservés mais pourraient nécessiter des ajustements pour afficher en fonction de mu)
        from xvfbwrapper import Xvfb
        import pyvista as pv
        import matplotlib.pyplot as plt

        vdisplay = Xvfb()
        vdisplay.start()
        pv.set_jupyter_backend('static')

        def pv_get_mesh(mesh_path):
            reader = pv.get_reader(mesh_path)
            mesh = reader.read()
            return mesh

        def pv_plot(mesh, scalars, title, clim=None, cmap=custom_cmap, cpos='xy', show_scalar_bar=True, show_edges=True):
            pl = pv.Plotter()
            pl.add_mesh(mesh, scalars=scalars, cmap=cmap, clim=clim, show_scalar_bar=show_scalar_bar, show_edges=show_edges)
            pl.add_title(title, font_size=12)
            pl.show(cpos=cpos)

        def myplots(title, scalars, err, clim, dim=2, field=f"cfpdes.parametric_diffusion.{name}", factor=1, cmap=custom_cmap): # Nom du champ modifié
            mesh = pv_get_mesh((f"cfpdes-{self.dim}d-p{self.order}.exports/Export.case"))
            pl = pv.Plotter(shape=(1,3))

            pl.subplot(0,0)
            pl.add_title(f'u = {title} ', font_size=8)
            pl.add_mesh(mesh[0].copy(), scalars=scalars, cmap=cmap, clim=clim, show_scalar_bar= False)
            pl.add_scalar_bar(title=f'u = {title} ', n_labels=4, label_font_size=12, title_font_size=14, fmt="%.2e", vertical=False, width=0.8, height=0.08, position_x=0.12, position_y=0.1)

            pl.subplot(0,1)
            pl.add_title('u_exact :', font_size=8)
            pl.add_mesh(mesh[0].copy(), scalars='cfpdes.expr.u_exact', cmap=custom_cmap, show_scalar_bar= False)
            pl.add_scalar_bar(title='u_ex', n_labels=4, label_font_size=12, title_font_size=14, fmt="%.2e", vertical=False, width=0.8, height=0.08, position_x=0.12, position_y=0.1)

            clim_err = [np.min(err), np.max(err)]
            pl.subplot(0,2)
            pl.add_title(f'|u_exact - u|: ', font_size=8)
            pl.add_mesh(mesh[0].copy(), scalars=err, cmap=custom_cmap, clim=clim_err, show_scalar_bar= False)
            pl.add_scalar_bar(title='u_ex-u/u_ex', n_labels=4, label_font_size=12, title_font_size=14, fmt="%.2e", vertical=False, width=0.8, height=0.08, position_x=0.12, position_y=0.1)

            pl.link_views()
            pl.view_xy()
            if plot == 1:
                pl.show()
                pl.screenshot(plot)

        if plot == 1:
           field = f"cfpdes.parametric_diffusion.u"
           mesh = pv_get_mesh(f"cfpdes-{self.dim}d-p{self.order}.exports/Export.case")
           pv_plot(mesh, field, title='Solution Feel++') # Affichage de la solution Feel++ seule

    # Affichage comparatif avec la solution exacte et l'erreur
           myplots(
               title='Feel++ Solution',
               scalars=field,
               err=np.abs(mesh[0].point_data['cfpdes.expr.u_exact'] - mesh[0].point_data[field]),
               clim=[np.min(mesh[0].point_data[field]), np.max(mesh[0].point_data[field])]
            )

        # Comparing solutions (conservé mais potentiellement à adapter si SciMBA est utilisé différemment pour le problème paramétrique)
        if has_scimba and solver == 'scimba':
            import pyvista as pv
            import torch

            u_scimba = self.scimba_solver( h=h, shape=shape, dim=self.dim, verbose=True)

            # File path to the .case file
            file_path = 'cfpdes-2d-p1.exports/Export.case'

            # Read the .case file using PyVista
            data = pv.read(file_path)

            # Iterate over each block in the dataset to find coordinates
            coordinates = None
            for i, block in enumerate(data):
                if block is None:
                    continue
                # Extract the mesh points (coordinates)
                coordinates = block.points
                solution = 'cfpdes.parametric_diffusion.u' # Nom du champ modifié

                df = pd.DataFrame(block.point_data)
                print(df.head())

            # Considering only 2d problems:
            num_features = coordinates.shape[1]
            print(f"Number of features in coordinates: {num_features}")
            if num_features > 2:
                coordinates = coordinates[:, :2]

            print(f"Number of points: {len(coordinates)}")
            print("\nNodes from export.case:", coordinates)

            feel_solution = block.point_data['cfpdes.parametric_diffusion.u'] # Nom du champ modifié
            u_ex = block.point_data['cfpdes.expr.u_exact']

            print("\nFeel++ solution 'cfpdes.parametric_diffusion.u':") # Nom du champ modifié
            print(feel_solution)

            coordinates_tensor = torch.tensor(coordinates, dtype=torch.float64, requires_grad=True)
            print(f"Shape of input tensor (coordinates): {coordinates_tensor.shape}")

            points = coordinates_tensor
            labels = torch.zeros(len(points))
            data = domain.SpaceTensor(points, labels, boundary=True)
            mu_tensor = torch.ones((len(points), 1), dtype=torch.float64, requires_grad=True) * self.mu_value # Utilisation de la valeur de mu courante

            scimba_solution = []

            u_values = u_scimba(data, mu_tensor) # Passage du tenseur mu

            for point, u_value in zip(points, u_values):
                print(f"u( {point[0:]} ) = {u_value[0]}")
                u_value_np = u_value.detach().numpy()

                scimba_solution = np.append(scimba_solution, u_value_np[0])

            print(f"ScimBa solution: {scimba_solution}")
            print(f"Feel++ solution: {feel_solution}")
            print(f"Exact solution: {u_ex}")
            print("\n Difference |scimba_solution - feel_solution| : ", np.abs(scimba_solution - feel_solution))

            # Plotting the solutions
            mesh = pv_get_mesh((f"cfpdes-{self.dim}d-p{self.order}.exports/Export.case"))

            # Plot the mesh with the chosen scalars and titles
            # Feel++ solution
            clim_feel = [np.min(feel_solution), np.max(feel_solution)]
            print('clim feel = ', clim_feel)
            pv_plot(mesh[0].copy(), feel_solution, title=f'Feel++ Solution (mu={self.mu_value}) \n clim = {clim_feel} ', clim=clim_feel) # Ajout de la valeur de mu au titre

            # ScimBa solution
            clim_scimba = [np.min(scimba_solution), np.max(scimba_solution)]
            print('clim scimba = ', clim_scimba)
            pv_plot(mesh[0].copy(), scimba_solution, title=f'Scimba Solution (mu={self.mu_value}) \n clim = {clim_scimba}', clim=clim_scimba) # Ajout de la valeur de mu au titre

            # Exact solution
            clim_exact = [np.min(u_ex), np.max(u_ex)]
            print('clim exact = ', clim_exact)
            pv_plot(mesh[0].copy(), u_ex, title=f'Exact Solution (mu={self.mu_value}) \n clim = {clim_exact}', clim=clim_exact) # Ajout de la valeur de mu au titre

            # Error plots
            err_feel = np.abs(u_ex - feel_solution) / np.abs(u_ex)
            err_scimba = np.abs(u_ex - scimba_solution) / np.abs(u_ex)
            clim_err = [np.min(err_feel), np.max(err_feel)]
            print('clim err_feel = |u_feel - u_exact|/|u_exact| ∈ ', clim_err)

            pv_plot(mesh[0].copy(), err_feel, title=f'|u_exact - u_feel|/|u_exact| (mu={self.mu_value}) \n clim = {clim_err}', clim=clim_err) # Ajout de la valeur de mu au titre
            print('clim err_scimba = |u_scimba - u_exact|/|u_exact| ∈ ', [np.min(err_scimba), np.max(err_scimba)])
            pv_plot(mesh[0].copy(), err_scimba, title=f'|u_exact - u_scimba|/|u_exact| (mu={self.mu_value}) \n clim = {[np.min(err_scimba), np.max(err_scimba)]}', clim=[np.min(err_scimba), np.max(err_scimba)]) # Ajout de la valeur de mu au titre

            myplots(title = 'u_feel : ', scalars='cfpdes.parametric_diffusion.u', err=np.abs(u_ex - feel_solution), clim=[np.min(feel_solution), np.max(feel_solution)]) # Nom du champ modifié
            print(' ||u_feel - u_exact||∞ = ', np.linalg.norm(feel_solution - u_ex, np.inf))

            myplots(title = 'u_scimba :', scalars=scimba_solution, err=np.abs(u_ex - scimba_solution), clim=[np.min(scimba_solution), np.max(scimba_solution)])
            print(' ||u_scimba - u_exact||∞ = ', np.linalg.norm(scimba_solution - u_ex, np.inf))

#______________________________________________________________________________________________

def runLaplacianPk(P, df, model, measures, verbose=False): # Garde le nom runLaplacianPk car la structure de convergence pourrait être réutilisée
    """Generate the Pk case"""
    meas = dict()
    dim, order, json = model
    for i, h in enumerate(df['h']):
        m= measures[i]  #feel_solver(filename=fn, h=h, json=json, dim=dim, verbose=verbose)
        print('measure =' , m)
        for norm in ['L2', 'H1']:
            meas.setdefault(f'P{order}-Norm_parametric_diffusion_{norm}-error', []) # Nom de la mesure modifié
            meas[f'P{order}-Norm_parametric_diffusion_{norm}-error'].append(m.get(f'Norm_parametric_diffusion_{norm}-error')) # Nom de la mesure modifié
    df = df.assign(**meas)
    for norm in ['L2', 'H1']:
        df[f'P{order}-parametric_diffusion_{norm}-convergence-rate']=np.log2(df[f'P{order}-Norm_parametric_diffusion_{norm}-error'].shift() / df[f'P{order}-Norm_parametric_diffusion_{norm}-error']) / np.log2(df['h'].shift() / df['h']) # Nom de la mesure modifié

    return df

def runConvergenceAnalysis(P, json, measures, dim=2,hs=[0.1, 0.05, 0.025, 0.0125],orders=[1],verbose=False): # Garde le nom runConvergenceAnalysis
    df=pd.DataFrame({'h':hs})
    for order in orders:
        df=runLaplacianPk(P, df=df,model=[dim,order,json(dim=dim,order=order)], measures = measures,verbose=verbose)
    print('df = ', df.to_markdown())
    return df

def plot_convergence(P, df,dim,orders=[1]): # Garde le nom plot_convergence
    fig=px.line(df, x="h", y=[f'P{order}-Norm_parametric_diffusion_{norm}-error' for order,norm in list(itertools.product(orders,['L2','H1']))]) # Nom de la mesure modifié
    fig.update_xaxes(title_text="h",type="log")
    fig.update_yaxes(title_text="Error",type="log")
    for order,norm in list(itertools.product(orders,['L2','H1'])):
        fig.update_traces(name=f'P{order} - {norm} error - {df[f"P{order}-parametric_diffusion_{norm}-convergence-rate"].iloc[-1]:.2f}', selector=dict(name=f'P{order}-Norm_parametric_diffusion_{norm}-error')) # Nom de la mesure modifié
    fig.update_layout(
            title=f"Convergence rate for the {dim}D Parametric Diffusion problem", # Titre modifié
            autosize=False,
            width=900,
            height=900,
        )
    return fig
#______________________________________________________________________________________________

# Définir les couleurs du bas au haut de la colormap de l'image
colors = [
    (0.188, 0.188, 0.220),   # bleu-noir
    #(75/255, 0, 130/255),    # indigo
    (0, 0, 255/255),        # bleu
    (0, 255/255, 255/255),  # cyan
    (0, 255/255, 0),        # vert
    (255/255, 255/255, 0),  # jaune
    (255/255, 127/255, 0),  # orange
    (255/255, 0 , 0),       # rouge
    (148/255, 0, 211/255),  # violet
    (1, 1, 1)                # blanc

]

cmap_name = 'custom_gradient'

# Créer la colormap
custom_cmap = LinearSegmentedColormap.from_list(cmap_name, colors, N=256)


# Afficher la colormap dans Matplotlib
plt.imshow(np.linspace(0, 1, 256)[None, :], aspect='auto', cmap=custom_cmap)
plt.colorbar()
plt.show()
#______________________________________________________________________________________________