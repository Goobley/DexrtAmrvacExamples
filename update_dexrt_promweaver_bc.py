import argparse
from pathlib import Path
import xarray as xr
import numpy as np
import lightweaver as lw
from lightweaver.rh_atoms import H_atom
import promweaver as pw

DTYPE = np.float32

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="preprocess_3d_cube",
        description="Update the promweaver boundary inside a dexrt input atmosphere"
    )
    parser.add_argument(
        "--path",
        dest="cube_path",
        help="Input file",
        metavar="FILE",
        type=Path
    )
    parser.add_argument(
        "--output",
        dest="output_path",
        help="Output file -- must be different to input",
        metavar="FILE",
        type=Path
    )

    args = parser.parse_args()

    ds = xr.open_dataset(args.cube_path)
    ds = ds.drop_vars([
        "prom_bc_I",
        "prom_bc_wavelength",
        "prom_bc_mu_min",
        "prom_bc_mu_max",
    ])

    model_atoms = pw.default_atomic_models()
    for i, a in enumerate(model_atoms):
        if a.element.name == "H":
            model_atoms[i] = H_atom()

    bc_ctx = pw.compute_falc_bc_ctx(active_atoms=["H", "Ca", "Mg"], atomic_models=model_atoms, prd=True)

    tabulated = pw.tabulate_bc(bc_ctx, mu_grid=np.linspace(0.05, 1.0, 20))
    I_with_zero = np.zeros((tabulated["I"].shape[0], tabulated["I"].shape[1] + 1))
    I_with_zero[:, 1:] = tabulated["I"][...]
    tabulated["I"] = I_with_zero
    tabulated["mu_grid"] = np.concatenate([[0], tabulated["mu_grid"]])

    bc_wavelength = tabulated["wavelength"].astype(DTYPE)
    bc_I = np.zeros(I_with_zero.shape, dtype=DTYPE)
    for la in range(tabulated["wavelength"].shape[0]):
        bc_I[la, :] = lw.convert_specific_intensity(
            tabulated["wavelength"][la],
            tabulated["I"][la, :],
            outUnits="kW / (m2 nm sr)"
        ).value

    bc_params = dict(
        prom_bc_I=(["prom_bc_wavelength", "prom_bc_mu"], bc_I),
        prom_bc_wavelength=(["prom_bc_wavelength"], bc_wavelength),
        prom_bc_mu_min=np.array([tabulated["mu_grid"][0]], dtype=DTYPE),
        prom_bc_mu_max=np.array([tabulated["mu_grid"][-1]], dtype=DTYPE),
    )
    ds = ds.update(bc_params)

    ds.to_netcdf(args.output_path)
