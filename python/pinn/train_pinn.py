from pathlib import Path
from scimba.equations import domain
from scimba.nets import training, training_tools
from scimba.pinns import pinn_losses, pinn_x, training_x
from scimba.sampling import sampling_parameters, sampling_pde, uniform_sampling

from poisson_square_2d import PoissonSquare2D

def main():
    xdomain = domain.SpaceDomain(2, domain.SquareDomain(2, [[0.0, 1.0], [0.0, 1.0]]))
    pde = PoissonSquare2D(xdomain)

    x_sampler = sampling_pde.XSampler(pde)
    mu_sampler = sampling_parameters.MuSampler(sampler=uniform_sampling.UniformSampling, model=pde)
    sampler = sampling_pde.PdeXCartesianSampler(x_sampler, mu_sampler)

    file_name = "poisson_square.pth"
    new_training = True

    # Remove the existing network file (if it exists)
    (Path.cwd() / Path(training_x.TrainerPINNSpace.FOLDER_FOR_SAVED_NETWORKS) / file_name).unlink(missing_ok=True)

    tlayers = [20, 20, 20, 20, 20]
    network = pinn_x.MLP_x(pde=pde, layer_sizes=tlayers, activation_type="sine")
    pinn = pinn_x.PINNx(network, pde)

    losses = pinn_losses.PinnLossesData(
        bc_loss_bool=True, 
        w_res=0.5, 
        w_bc=10, 
        residual_f_loss=training.MassLoss()
    )
    optimizers = training_tools.OptimizerData(learning_rate=1.0e-2, decay=0.99)

    trainer = training_x.TrainerPINNSpace(
        pde=pde,
        network=pinn,
        losses=losses,
        optimizers=optimizers,
        sampler=sampler,
        file_name=file_name,
        batch_size=4000,
    )

    if new_training:
        trainer.train(epochs=1200, n_collocation=4000, n_bc_collocation=2000, n_data=0)

    trainer.plot(20000, reference_solution=True)

if __name__ == "__main__":
    main()
