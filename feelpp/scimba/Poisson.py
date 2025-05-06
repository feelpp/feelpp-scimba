import os
import feelpp.core as fppc
from feelpp.toolboxes.cfpdes import *
from feelpp.scimba.scimba_pinns import Run_Poisson2D, Poisson_2D, PoissonDisk2D
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
  
##______________________________________________________________________________________________

  def feel_solver(self, filename, json, h, shape = 'Rectangle',verbose=False):
    if verbose:
      print(f"Solving the laplacian problem for h = {h}...")
    poisson = self.pb
    poisson.setMesh(self.getMesh(f"omega-{self.dim}.geo", h=h, shape =shape, verbose=verbose))
    poisson.setModelProperties(json)
    poisson.init(buildModelAlgebraicFactory=True)
    poisson.printAndSaveInfo()
    poisson.solve()
    poisson.exportResults()
    measures = poisson.postProcessMeasures().values()
    return measures
  
##______________________________________________________________________________________________

  def scimba_solver(self, h, shape='Rectangle', dim = 2, verbose=False):
    if verbose:
      print(f"Solving a Poisson problem for h = {h}...")    
    
    diff = self.diff.replace('{', '(').replace('}', ')')
    
    if shape == 'Disk':
      xdomain = domain.SpaceDomain(2, domain.DiskBasedDomain(2, center=[0.0, 0.0], radius=1.0))
  
    elif shape == 'Rectangle':
      xdomain = domain.SpaceDomain(2, domain.SquareDomain(2, [[0.0, 1.0], [0.0, 1.0]]))
    
    pde = Poisson_2D(xdomain, rhs=self.rhs, diff=diff, g=self.g, u_exact=self.u_exact)
    u , pinn = Run_Poisson2D(pde, epoch=1000)

    return u

##______________________________________________________________________________________________
  
  def __call__(self,
               h=0.05,                                      # mesh size 
               order=1,                                     # polynomial order 
               name='u',                                    # name of the variable u
               rhs='8*pi*pi*sin(2*pi*x)*sin(2*pi*y)',       # right hand side
               diff='{1,0,0,1}',                            # diffusion matrix
               g='0',                                       # Dirichlet boundary conditions
               gN='0',                                      # Neumann boundary conditions
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
    a = 0.0
    self.h = h
    self.measures = dict()
    self.rhs = rhs
    self.g = g
    self.gN = gN
    self.u_exact = u_exact
    self.diff = diff
    self.pb = cfpdes(dim=self.dim, keyword=f"cfpdes-{self.dim}d-p{self.order}")
    self.model = lambda order,dim=2,name="u": {
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
              "f": f"{rhs}:x:y"  if self.dim == 2 else f"{rhs}:x:y:z",
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
        "poisson":
        {
          "Dirichlet":
          {
            "g":
            {
              "markers":["Gamma_D"],
              "expr":f"{g}:x:y" if self.dim == 2 else f"{g}:x:y:z"
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
      self.genGeometry(fn, self.h, shape=shape)
    else:
      fn = geofile    
##________________________

  # Solving

    poisson_json = self.model
    self.measures = self.feel_solver(filename=fn, h=self.h, shape =shape, json=poisson_json(order=self.order,dim=self.dim), verbose=True)

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
    
    def pv_plot(mesh, scalars, title, clim=None, cmap=custom_cmap, cpos='xy', show_scalar_bar=True, show_edges=True):
      pl = pv.Plotter()
      pl.add_mesh(mesh, scalars=scalars, cmap=cmap, clim=clim, show_scalar_bar=show_scalar_bar, show_edges=show_edges)
      pl.add_title(title, font_size=12)
      pl.show(cpos=cpos)

#    def pv_plot(mesh, field, clim=None, cmap=custom_cmap, cpos='xy', show_scalar_bar=True, show_edges=True):
#      mesh.plot(scalars=field, cmap=cmap, cpos=cpos, show_scalar_bar=show_scalar_bar, show_edges=show_edges)

    def myplots(title, scalars, err, clim, dim=2, field=f"cfpdes.poisson.{name}", factor=1, cmap=custom_cmap):
      mesh = pv_get_mesh((f"cfpdes-{self.dim}d-p{self.order}.exports/Export.case"))
      pl = pv.Plotter(shape=(1,3))

      pl.subplot(0,0)
      pl.add_title(f'u = {title} ', font_size=8)
      pl.add_mesh(mesh[0].copy(), scalars=scalars, cmap=cmap, clim=clim, show_scalar_bar= False)
      pl.add_scalar_bar(
            title=f'u = {title} ',
            n_labels=4,  # Reduce the number of labels for clarity
            label_font_size=12,  # Increase label font size for better readability
            title_font_size=14,  # Increase title font size for better readability
            fmt="%.2e",  # Scientific notation format
            vertical=False,
            width=0.8,  # Increase width
            height=0.08,  # Maintain height
            position_x=0.12,  # Adjust x position
            position_y=0.1  # Raise the y position
            )


      pl.subplot(0,1)
      pl.add_title('u_exact :', font_size=8)      
      pl.add_mesh(mesh[0].copy(), scalars='cfpdes.expr.u_exact', cmap=custom_cmap, show_scalar_bar= False)
      pl.add_scalar_bar(
            title='u_ex',
            n_labels=4,  # Reduce the number of labels for clarity
            label_font_size=12,  # Increase label font size for better readability
            title_font_size=14,  # Increase title font size for better readability
            fmt="%.2e",  # Scientific notation format
            vertical=False,
            width=0.8,  # Increase width
            height=0.08,  # Maintain height
            position_x=0.12,  # Adjust x position
            position_y=0.1  # Raise the y position
            )

      clim_err = [np.min(err), np.max(err)]
      pl.subplot(0,2)
      pl.add_title(f'|u_exact - u|: ', font_size=8)
      pl.add_mesh(mesh[0].copy(), scalars=err, cmap=custom_cmap, clim=clim_err, show_scalar_bar= False)
      pl.add_scalar_bar(
            title='u_ex-u/u_ex',
            n_labels=4,  # Reduce the number of labels for clarity
            label_font_size=12,  # Increase label font size for better readability
            title_font_size=14,  # Increase title font size for better readability
            fmt="%.2e",  # Scientific notation format
            vertical=False,
            width=0.8,  # Increase width
            height=0.08,  # Maintain height
            position_x=0.12,  # Adjust x position
            position_y=0.1  # Raise the y position
            )
      
      pl.link_views()
      pl.view_xy()    
      if plot == 1:
        pl.show()
        pl.screenshot(plot)

#_____

    if plot == 1:
      field="cfpdes.poisson.u"
      mesh = pv_get_mesh(f"cfpdes-{self.dim}d-p{self.order}.exports/Export.case")
      pv_plot(mesh, field,title='Solution')

      #myplots(dim=2,factor=1)

    # Comparing solutions
    if solver == 'scimba':
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
        solution = 'cfpdes.poisson.u'

        df = pd.DataFrame(block.point_data)
        print(df.head())

      # Considering only 2d problems:
      num_features = coordinates.shape[1]
      print(f"Number of features in coordinates: {num_features}")
      if num_features > 2:
        coordinates = coordinates[:, :2]

      print(f"Number of points: {len(coordinates)}")          
      print("\nNodes from export.case:", coordinates)

      feel_solution = block.point_data['cfpdes.poisson.u']
      u_ex = block.point_data['cfpdes.expr.u_exact']
      
      print("\nFeel++ solution 'cfpdes.poisson.u':")
      print(feel_solution) 

      coordinates_tensor = torch.tensor(coordinates, dtype=torch.float64, requires_grad=True)
      print(f"Shape of input tensor (coordinates): {coordinates_tensor.shape}")
      
      points = coordinates_tensor
      labels = torch.zeros(len(points))
      data = domain.SpaceTensor(points, labels, boundary=True)
      mu = torch.ones((len(points), 1), dtype=torch.float64, requires_grad=True)

      scimba_solution = []
      
      u_values = u_scimba(data, mu)

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
      pv_plot(mesh[0].copy(), feel_solution, title=f'Feel++ Solution \n clim = {clim_feel} ', clim=clim_feel)

      # ScimBa solution
      clim_scimba = [np.min(scimba_solution), np.max(scimba_solution)]
      print('clim scimba = ', clim_scimba)
      pv_plot(mesh[0].copy(), scimba_solution, title=f'Scimba Solution \n clim = {clim_scimba}', clim=clim_scimba)
      
      # Exact solution
      clim_exact = [np.min(u_ex), np.max(u_ex)]
      print('clim exact = ', clim_exact)
      pv_plot(mesh[0].copy(), u_ex, title=f'Exact Solution \n clim = {clim_exact}', clim=clim_exact)

      # Error plots
      err_feel = np.abs(u_ex - feel_solution) / np.abs(u_ex)
      err_scimba = np.abs(u_ex - scimba_solution) / np.abs(u_ex)
      self.errl2_scimba = np.sqrt(np.sum((scimba_solution - u_ex)**2) / np.sum(u_ex**2))

      clim_err = [np.min(err_feel), np.max(err_feel)]
      print('clim err_feel = |u_feel - u_exact|/|u_exact| ∈ ', clim_err)

      pv_plot(mesh[0].copy(), err_feel, title=f'|u_exact - u_feel|/|u_exact| \n clim = {clim_err}', clim=clim_err)
      print('clim err_scimba = |u_scimba - u_exact|/|u_exact| ∈ ', [np.min(err_scimba), np.max(err_scimba)])
      pv_plot(mesh[0].copy(), err_scimba, title=f'|u_exact - u_scimba|/|u_exact| \n clim = {[np.min(err_scimba), np.max(err_scimba)]}', clim=[np.min(err_scimba), np.max(err_scimba)])

      myplots(title = 'u_feel : ', scalars='cfpdes.poisson.u', err=np.abs(u_ex - feel_solution), clim=[np.min(feel_solution), np.max(feel_solution)])
      print(' ||u_feel - u_exact||∞ = ', np.linalg.norm(feel_solution - u_ex, np.inf))

      myplots(title = 'u_scimba :', scalars=scimba_solution, err=np.abs(u_ex - scimba_solution), clim=[np.min(scimba_solution), np.max(scimba_solution)])
      print(' ||u_scimba - u_exact||∞ = ', np.linalg.norm(scimba_solution - u_ex, np.inf))

      """
      # Plotting the solutions


      pl = pv.Plotter(shape=(3, 2))

      # First row: u_feel and u_scimba

      # u_feel plot
      clim_feel = [np.min(feel_solution), np.max(feel_solution)]
      print('clim feel = ', clim_feel)
      pl.subplot(0, 0)
      pl.add_title('u_feel', font_size=8)
      pl.add_mesh(mesh[0].copy(), scalars=feel_solution, cmap=custom_cmap, clim=clim_feel)
      pl.add_scalar_bar(title='u_feel',
                        n_labels=5,
                        label_font_size=12,
                        title_font_size=14,
                        fmt="%.2e",
                        vertical=False,
                        width=0.5,
                        height=0.08,
                        position_x=0.35,
                        position_y=0.02)

      # u_scimba plot
      clim_scimba = [np.min(scimba_solution), np.max(scimba_solution)]
      print('clim scimba = ', clim_scimba)
      pl.subplot(0, 1)
      pl.add_title('u_scimba', font_size=8)
      pl.add_mesh(mesh[0].copy(), scalars=scimba_solution, cmap=custom_cmap, clim=clim_scimba)
      pl.add_scalar_bar(title='u_scimba',
                        n_labels=5,
                        label_font_size=12,
                        title_font_size=14,
                        fmt="%.2e",
                        vertical=False,
                        width=0.5,
                        height=0.08,
                        position_x=0.35,
                        position_y=0.02)

      # Second row: u_exact and u_scimba - u_feel (normalized)

      # u_exact plot
      clim_exact = [np.min(u_ex), np.max(u_ex)]
      print('clim u_ex = ', clim_exact)
      pl.subplot(1, 0)
      pl.add_title('u_exact', font_size=8)
      pl.add_mesh(mesh[0].copy(), scalars=u_ex, cmap=custom_cmap, clim=clim_exact)
      pl.add_scalar_bar(title='u_exact',
                        n_labels=5,
                        label_font_size=12,
                        title_font_size=14,
                        fmt="%.2e",
                        vertical=False,
                        width=0.5,
                        height=0.08,
                        position_x=0.35,
                        position_y=0.02)

      # |u_scimba - u_feel|/|u_exact| plot
      err_scimba_feel = np.abs(scimba_solution - feel_solution) / np.abs(u_ex)
      print('err = ', err_scimba_feel)
      clim_err_scimba_feel = [np.min(err_scimba_feel), np.max(err_scimba_feel)]
      print('clim u_sc - u_feel =  = ', clim_err_scimba_feel)
      pl.subplot(1, 1)
      pl.add_title('|u_scimba - u_feel|/|u_exact|', font_size=8)
      pl.add_mesh(mesh[0].copy(), scalars=err_scimba_feel, cmap=custom_cmap, clim=clim_err_scimba_feel)
      pl.add_scalar_bar(title='|u_scimba - u_feel|/|u_exact|',
                        n_labels=5,
                        label_font_size=12,
                        title_font_size=14,
                        fmt="%.2e",
                        vertical=False,
                        width=0.5,
                        height=0.08,
                        position_x=0.35,
                        position_y=0.02)

      print(' ||u_scimba - u_feel||∞ = ', np.linalg.norm(scimba_solution - feel_solution, np.inf))

      # Third row: |u_exact - u_scimba|/|u_exact| and |u_exact - u_feel|/|u_exact|

      # |u_exact - u_scimba|/|u_exact| plot
      err_exact_scimba = np.abs(u_ex - scimba_solution) / np.abs(u_ex)
      print('err = ', err_exact_scimba)
      clim_err_exact_scimba = [np.min(err_exact_scimba), np.max(err_exact_scimba)]
      print('clim ex - sc = ', clim_err_exact_scimba)
      pl.subplot(2, 0)
      pl.add_title('|u_exact - u_scimba|/|u_exact|', font_size=8)
      pl.add_mesh(mesh[0].copy(), scalars=err_exact_scimba, cmap=custom_cmap, clim=clim_err_exact_scimba)
      pl.add_scalar_bar(title='u_exact - u_scimba',
                        n_labels=5,
                        label_font_size=12,
                        title_font_size=14,
                        fmt="%.2e",
                        vertical=False,
                        width=0.5,
                        height=0.08,
                        position_x=0.35,
                        position_y=0.02)

      # |u_exact - u_feel|/|u_exact| plot
      err_exact_feel = np.abs(u_ex - feel_solution) / np.abs(u_ex)
      print('err = ', err_exact_feel)
      clim_err_exact_feel = [np.min(err_exact_feel), np.max(err_exact_feel)]
      print('clim ex - feel = ', clim_err_exact_feel)
      pl.subplot(2, 1)
      pl.add_title('|u_exact - u_feel|/|u_exact|', font_size=8)
      pl.add_mesh(mesh[0].copy(), scalars=err_exact_feel, cmap=custom_cmap, clim=clim_err_exact_feel)
      pl.add_scalar_bar(title='u_exact - u_feel',
                        n_labels=5,
                        label_font_size=12,
                        title_font_size=14,
                        fmt="%.2e",
                        vertical=False,
                        width=0.5,
                        height=0.08,
                        position_x=0.35,
                        position_y=0.02)

      pl.link_views()
      pl.view_xy()
      if plot == 1:
          pl.show()
          pl.screenshot(plot)
      """

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
    (0.188, 0.188, 0.220),  # bleu-noir
    #(75/255, 0, 130/255),   # indigo
    (0, 0, 255/255),        # bleu
    (0, 255/255, 255/255),  # cyan
    (0, 255/255, 0),        # vert
    (255/255, 255/255, 0),  # jaune
    (255/255, 127/255, 0),  # orange
    (255/255, 0 , 0),       # rouge
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