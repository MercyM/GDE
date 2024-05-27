from .facmac_learner import FACMACLearner
from .facmac_learner_discrete import FACMACDiscreteLearner
from .EA_facmac_learner_discrete import EA_FACMACDiscreteLearner
REGISTRY = {}
REGISTRY["facmac_learner"] = FACMACLearner
REGISTRY["facmac_learner_discrete"] = FACMACDiscreteLearner
REGISTRY["facmac_learner_discrete_EA"] = EA_FACMACDiscreteLearner