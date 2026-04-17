import h5py
import numpy as np


class ICsWriter:
    def __init__(self, dm_info=None, gas_info=None, star_info=None):
        self.dm_info = dm_info
        self.gas_info = gas_info
        self.star_info = star_info

        self._NumPart = np.zeros(6, dtype=np.int32)

        if self.dm_info is not None:
            self._NumPart[1] = self.dm_info["part_coords"].shape[0]
        if self.gas_info is not None:
            self._NumPart[0] = self.gas_info["cell_coords"].shape[0]
        if self.star_info is not None:
            self._NumPart[4] = self.star_info["part_coords"].shape[0]

    def _write_header(self, ICs):
        _header = ICs.create_group("Header")

        _header.attrs.create("NumPart_ThisFile", self._NumPart)
        _header.attrs.create("NumPart_Total", self._NumPart)
        _header.attrs.create("NumPart_Total_HighWord", np.zeros(6, dtype=np.int32))
        _header.attrs.create("MassTable", np.zeros(6, dtype=np.float64))

        _header.attrs.create("Time", 0.0)
        _header.attrs.create("Redshift", 0.0)
        _header.attrs.create("BoxSize", 0.0)
        _header.attrs.create("NumFilesPerSnapshot", 1)

        _header.attrs.create("Omega0", 0.0)
        _header.attrs.create("OmegaB", 0.0)
        _header.attrs.create("OmegaLambda", 0.0)
        _header.attrs.create("HubbleParam", 1.0)

        _header.attrs.create("Flag_Sfr", 0)
        _header.attrs.create("Flag_Cooling", 0)
        _header.attrs.create("Flag_StellarAge", 0)
        _header.attrs.create("Flag_Metals", 0)
        _header.attrs.create("Flag_Feedback", 0)

        _header.attrs.create("Flag_DoublePrecision", 1)

    def _write_dm(self, ICs, dm_info):
        _part1 = ICs.create_group("PartType1")

        _part1.create_dataset("ParticleIDs", data=dm_info["part_ids"])
        _part1.create_dataset("Coordinates", data=dm_info["part_coords"])
        _part1.create_dataset("Masses", data=dm_info["part_masses"])
        _part1.create_dataset("Velocities", data=dm_info["part_velocs"])

    def _write_gas(self, ICs, gas_info):
        _part0 = ICs.create_group("PartType0")

        _part0.create_dataset("ParticleIDs", data=gas_info["cell_ids"])
        _part0.create_dataset("Coordinates", data=gas_info["cell_coords"])
        _part0.create_dataset("Masses", data=gas_info["cell_masses"])
        _part0.create_dataset("Velocities", data=gas_info["cell_velocs"])
        _part0.create_dataset("InternalEnergy", data=gas_info["cell_internal_energies"])

    def _write_star(self, ICs, star_info):
        _part4 = ICs.create_group("PartType4")

        _part4.create_dataset("ParticleIDs", data=star_info["part_ids"])
        _part4.create_dataset("Coordinates", data=star_info["part_coords"])
        _part4.create_dataset("Masses", data=star_info["part_masses"])
        _part4.create_dataset("Velocities", data=star_info["part_velocs"])

    def write(self, filename):
        with h5py.File(filename, "w") as ICs:
            self._write_header(ICs)
            if self.dm_info is not None:
                self._write_dm(ICs, self.dm_info)
            if self.gas_info is not None:
                self._write_gas(ICs, self.gas_info)
            if self.star_info is not None:
                self._write_star(ICs, self.star_info)
