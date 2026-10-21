


class RadOncGymError(Exception):
    pass


class ContractError(RadOncGymError, ValueError):
    pass


class EpisodeStateError(RadOncGymError, RuntimeError):
    pass


class BackendContractError(RadOncGymError, RuntimeError):
    pass

