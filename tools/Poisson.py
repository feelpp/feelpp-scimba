import os
import feelpp
from feelpp.toolboxes.cfpdes import *
from tools.scimba_pinns import Run_laplacian2D, Poisson_2D, PoissonDisk2D
from scimba.equations import domain

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


class Poisson:
  """
  Solves the problem
  -Laplacian u = f   in Omega
  u            = g   in boundary
  
  - with f,g are set by the user
  """
  def __init__(self, dim=2, order=1):

    self.dim   = dim
    self.model = dict()
    self.order = order  
##______________________________________________________________________________________________


  def genGeometry(self, filename, h=0.05, shape = 'Rectangle'):
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
        Physical Surface("Gamma_D") = {1};
        Physical Volume("Omega") = {1};
        """
    with open(filename, 'w') as f:
      f.write(geo)

  def getMesh(self, filename,h=0.05, shape = 'Rectangle',verbose=False):
    """create mesh"""
    import os
    for ext in [".msh",".geo"]:
        f=os.path.splitext(filename)[0]+ext
        if os.path.exists(f):
            os.remove(f)
    if verbose:
        print(f"generate mesh {filename} with h={h} and dimension={self.dim}")
    self.genGeometry(filename=filename,h=h, shape=shape)
    mesh = feelpp.load(feelpp.mesh(dim=self.dim,realdim=self.dim), filename, h)
    return mesh
  
##______________________________________________________________________________________________

  def feel_solver(self, filename, h, json,verbose=False):
    if verbose:
      print(f"Solving the laplacian problem for h = {h}...")
    poisson = self.pb
    poisson.setMesh(self.getMesh(f"omega-{self.dim}.geo",h=h,verbose=verbose))
    poisson.setModelProperties(json)
    poisson.init(buildModelAlgebraicFactory=True)
    poisson.printAndSaveInfo()
    poisson.solve()
    poisson.exportResults()
    measures = poisson.postProcessMeasures().values()
    return measures
  
##______________________________________________________________________________________________

  def scimba_solver(self, shape='Rectangle', h=0.05, dim = 2, verbose=False):
    if verbose:
      print(f"Solving a Poisson problem for h = {h}...")    
    
    diff = self.diff.replace('{', '(').replace('}', ')')
    
    if shape == 'disk':
      xdomain = domain.SpaceDomain(2, domain.DiskBasedDomain(2, center=[0.0, 0.0], radius=1.0))
  
    elif shape == 'Rectangle':
      xdomain = domain.SpaceDomain(2, domain.SquareDomain(2, [[0.0, 1.0], [0.0, 1.0]]))
    
    pde = Poisson_2D(xdomain, rhs=self.rhs, diff=diff, g=self.g, u_exact=self.u_exact)
    network, pde = Run_laplacian2D(pde)

    # Extract solution function u
    u = network.forward

    return u

##______________________________________________________________________________________________
  
  def __call__(self,
               h=0.05,                                      # mesh size 
               order=1,                                     # polynomial order 
               name='u',                                    # name of the variable u
               rhs='8*pi*pi*sin(2*pi*x)*sin(2*pi*y)',       # right hand side
               diff='{1,0,0,1}',                            # diffusion matrix
               g='0',                                       # boundary conditions
               shape='Rectangle',                           # domain shape (Rectangle, Disk)    
               geofile=None,                                # geometry file
               plot=1,                                      # plot the solution
               solver='feelpp',                             # solver 
               u_exact='sin(2 * pi * x) * sin(2 * pi * y)',
               grad_u_exact = '{2*pi*cos(2*pi*x)*sin(2*pi*y),2*pi*sin(2*pi*x)*cos(2*pi*y)}' 
               ):
    """
    Solves the problem where :
    - h is the mesh size
    - order the polynomial order
    - rhs is the expression of the right-hand side f(x,y)
    """
    self.measures = dict()
    self.rhs = rhs
    self.g = g
    self.u_exact = u_exact
    self.diff = diff
    self.pb    = cfpdes(dim=self.dim, keyword=f"cfpdes-{self.dim}d-p{self.order}")
    self.model = {
      "Name": "Poisson",
      "ShortName": "Poisson",
      "Models":
      {
        f"cfpdes-{self.dim}d-p{self.order}":
        {
          "equations":"poisson"
        },
        "poisson":{
          "setup":{
            "unknown":{
              "basis":f"Pch{order}",
              "name":f"{name}",
              "symbol":"u"
            },
            "coefficients":{
              "c": f"{diff}:x:y" if self.dim == 2 else f"{diff}:x:y:z",
              "f": f"{rhs}:x:y"  if self.dim == 2 else f"{rhs}:x:y:z"
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
        "poisson":
        {
          "Dirichlet":
          {
            "g":
            {
              "markers":["Gamma_D"],
              "expr":f"{g}:x:y"
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
              #"u_scimba": f"{u_scimba}:x:y" if self.dim == 2 else f"{name}:x:y:z",
              "rhs": f"{rhs}:x:y" if self.dim == 2 else f"{rhs}:x:y:z",
              "u_exact" : f"{u_exact}:x:y" if self.dim==2 else f"{u_exact}:x:y:z",
              "grad_u_exact" : f"{grad_u_exact}:x:y" if self.dim==2 else f"{grad_u_exact}:x:y:z"

            }
          },
            "Measures" :
            {
              "Norm" :
              {
                  "poisson" :
                  {
                     "type":["L2-error", "H1-error"],
                     "field":f"poisson.{name}",
                     "solution": f"{u_exact}:x:y" if self.dim==2 else f"{u_exact}:x:y:z",
                     "grad_solution": f"{grad_u_exact}:x:y" if self.dim==2 else f"{grad_u_exact}:x:y:z",
                     "markers":"Omega",
                     "quad":6
                 }
              },
                "Statistics":
                {
                    "mystatA":
                    {
                        "type":["min","max","mean","integrate"],
                        "field":f"poisson.{name}"
                    }
                }
            }
        }
      }
    }
    
    """
    fn = f'omega-{self.dim}.geo'
    self.genGeometry(fn, h=h, shape=geofile)
    """
    fn = None
    if geofile is None:
      fn = f'omega-{self.dim}.geo'
      self.genGeometry(fn, h, shape=shape)
    else:
      fn = geofile    
##________________________

  # Solving

    poisson_json = lambda order,dim=2,name="u": self.model
    self.measures = self.feel_solver(filename=fn, h=h, json=poisson_json(order=self.order,dim=self.dim), verbose=True)
        
    if solver == 'scimba':
      import pyvista as pv
      import torch
      from tools.GmeshRead import mesh2d

      u_scimba = self.scimba_solver(shape=shape, h=h, dim=self.dim, verbose=True)
      
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
        solution = 'cfpdes.poisson.u'
        solution_expression = block.point_data[solution]

        df = pd.DataFrame(block.point_data)
        print(df.head())


      # Considering only 2d problems:
      num_features = coordinates.shape[1]
      print(f"Number of features in coordinates: {num_features}")
      if num_features > 2:
        coordinates = coordinates[:, :2]

      print(f"Number of points: {len(coordinates)}")          
      print("\nNodes:"  , coordinates)
      feel_solution = block.point_data['cfpdes.poisson.u']
      print("\nFeel++ solution 'cfpdes.poisson.u':")
      print(feel_solution) 


      mesh = "omega-2.msh"
      my_mesh = mesh2d(mesh)
      my_mesh.read_mesh()
      print("Number of nodes:", my_mesh.Nnodes)
      print("Nodes coordinates:", my_mesh.Nodes)
      print('difference = ', coordinates - my_mesh.Nodes)

      # Convert coordinates to tensor
      #coordinates = my_mesh.Nodes
      coordinates_tensor = torch.tensor(coordinates, dtype=torch.double)
      print(f"Shape of input tensor (coordinates): {coordinates_tensor.shape}")
      
      # Create the mu tensor with correct shape
      mu_value = 1
      mu = torch.full((coordinates_tensor.size(0), 1), mu_value, dtype=torch.double)

      solution_tensor = u_scimba(coordinates_tensor, mu)
      solution_array = solution_tensor.detach().numpy().flatten()
     

      print(f"ScimBa solution: {solution_array}")
      print("\n Difference : ", solution_array - feel_solution)

##________________________   
    # Plots

    from xvfbwrapper import Xvfb
    import pyvista as pv 
    import matplotlib.pyplot as plt


    vdisplay = Xvfb()
    vdisplay.start()
    pv.set_jupyter_backend('static') 
    #pv.start_xvfb()
    def pv_get_mesh(mesh_path):
      reader = pv.get_reader(mesh_path)
      mesh = reader.read()
      return mesh

    def pv_plot(mesh, field, clim=None, cmap=custom_cmap, cpos='xy', show_scalar_bar=True, show_edges=True):
      mesh.plot(scalars=field, clim=clim, cmap=cmap, cpos=cpos, show_scalar_bar=show_scalar_bar, show_edges=show_edges)

    def myplots(dim=2, field=f"cfpdes.poisson.{name}", factor=1, cmap=custom_cmap):
      mesh = pv_get_mesh((f"cfpdes-{self.dim}d-p{self.order}.exports/Export.case"))
      #pv_plot(mesh, field)
      pl = pv.Plotter(shape=(1,2))
      if solver == 'scimba':
        pl = pv.Plotter(shape=(1,3))
        pl.subplot(0,2)
        pl.add_title('u_scimba ' , font_size=10)
        pl.add_mesh(mesh[0], scalars = solution_array, cmap=custom_cmap)

      pl.subplot(0,0)
      pl.add_title(f'Solution P{order}', font_size=10)
      pl.add_mesh(mesh[0].copy(), scalars = f"cfpdes.poisson.{name}", cmap=custom_cmap)

      pl.subplot(0,1)
      pl.add_title('u_exact', font_size=10)
      pl.add_mesh(mesh[0].copy(), scalars = 'cfpdes.expr.u_exact', cmap=custom_cmap)      

      pl.link_views()
      pl.view_xy()    
      if plot != None:
        pl.show()
        pl.screenshot(plot)

    myplots(dim=2,factor=0.5)

#______________________________________________________________________________________________

def runLaplacianPk(P, df, model, measures, verbose=False):
  """Generate the Pk case"""
  meas = dict()
  dim, order, json = model    
  for i, h in enumerate(df['h']):
    m= measures[i]   #feel_solver(filename=fn, h=h, json=json, dim=dim, verbose=verbose)
    print('measure =' , m)
    for norm in ['L2', 'H1']:
      meas.setdefault(f'P{order}-Norm_poisson_{norm}-error', [])
      meas[f'P{order}-Norm_poisson_{norm}-error'].append(m.get(f'Norm_poisson_{norm}-error'))
  df = df.assign(**meas)
  for norm in ['L2', 'H1']:
    df[f'P{order}-poisson_{norm}-convergence-rate']=np.log2(df[f'P{order}-Norm_poisson_{norm}-error'].shift() / df[f'P{order}-Norm_poisson_{norm}-error']) / np.log2(df['h'].shift() / df['h'])

  return df

def runConvergenceAnalysis(P, json, measures, dim=2,hs=[0.1, 0.05, 0.025, 0.0125],orders=[1],verbose=False):
  df=pd.DataFrame({'h':hs})
  for order in orders:
    df=runLaplacianPk(P, df=df,model=[dim,order,json(dim=dim,order=order)], measures = measures,verbose=verbose)
  print('df = ', df.to_markdown())
  return df

def plot_convergence(P, df,dim,orders=[1]):
  fig=px.line(df, x="h", y=[f'P{order}-Norm_poisson_{norm}-error' for order,norm in list(itertools.product(orders,['L2','H1']))])
  fig.update_xaxes(title_text="h",type="log")
  fig.update_yaxes(title_text="Error",type="log")
  for order,norm in list(itertools.product(orders,['L2','H1'])):
    fig.update_traces(name=f'P{order} - {norm} error - {df[f"P{order}-poisson_{norm}-convergence-rate"].iloc[-1]:.2f}', selector=dict(name=f'P{order}-Norm_poisson_{norm}-error'))
  fig.update_layout(
          title=f"Convergence rate for the {dim}D Poisson problem",
          autosize=False,
          width=900,
          height=900,
      )
  return fig
#______________________________________________________________________________________________

# Définir les couleurs du bas au haut de la colormap de l'image
colors = [
    (75/255, 0, 130/255),   # indigo
    (0, 0, 255/255),        # bleu
    (0, 255/255, 255/255),  # cyan
    (0, 255/255, 0),        # vert
    (255/255, 255/255, 0),  # jaune
    (255/255, 127/255, 0),  # orange
    (255/255, 0 , 0),        # rouge
    (148/255, 0, 211/255),  # violet
    (1, 1, 1)               # blanc

]

cmap_name = 'custom_gradient'

# Créer la colormap
custom_cmap = LinearSegmentedColormap.from_list(cmap_name, colors, N=256)


# Afficher la colormap dans Matplotlib
plt.imshow(np.linspace(0, 1, 256)[None, :], aspect='auto', cmap=custom_cmap)
plt.colorbar()
plt.show()
#______________________________________________________________________________________________
