import h5py
import numpy as np


class ICsWriter:
    def __init__(self, **kwargs):
        """
        Initialize the ICsWriter with the particle data.

        Parameters:
        ----------
        **kwargs: dict
            Dictionary of particle data for each particle type.
            The keys should be in the format "PartTypeX",
            where X is the particle type (0 for gas, 1 for dark matter, 2 for stars, etc.).
            The values should be dictionaries containing the particle data arrays,
            with keys "ParticleIDs", "Coordinates", "Velocities", and "Masses"
            (as well as any other relevant data, such as "InternalEnergy" for gas particles).
        """

        for parttype, part_data in kwargs.items():
            self.__setattr__(parttype, part_data)

    def _get_num_part(self):
        """
        Get the number of particles for each particle type.

        Returns:
        ----------
        NumPart: array
            Array of the number of particles for each particle type.
        """

        NumPart = np.zeros(6, dtype=np.int32)

        for i in range(6):
            parttype = f"PartType{i}"
            if parttype in self.__dict__:
                NumPart[i] = np.shape(self.__getattribute__(parttype)["Coordinates"])[0]

        return NumPart

    def _write_header(self, ICs):
        """
        Write the header information to the ICs file.
        """

        _header = ICs.create_group("Header")

        _NumPart = self._get_num_part()

        _header.attrs.create("NumPart_ThisFile", _NumPart)
        _header.attrs.create("NumPart_Total", _NumPart)
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

    def _write_parttype(self, ICs, parttype):
        """
        Write the particle data for a given particle type to the ICs file.
        """

        if parttype in self.__dict__:
            _part_data = ICs.create_group(parttype)

            for _key, _data_array in self.__getattribute__(parttype).items():
                _part_data.create_dataset(_key, data=_data_array)

    def write(self, filename):
        """
        Write the ICs file with the particle data.

        Parameters:
        ----------
        filename: str
            Path to the output ICs file.
        """

        with h5py.File(filename, "w") as ICs:
            self._write_header(ICs)

            for i in range(6):
                parttype = f"PartType{i}"
                self._write_parttype(ICs, parttype)
