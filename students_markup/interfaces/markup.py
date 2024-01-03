from abc import ABC
from abc import abstractmethod
from typing import Any
from typing import Dict
from typing import List
from typing import Optional
from typing import Tuple

import numpy as np


class IMarkup(ABC):
    @abstractmethod
    def markup(
        self,
        image: np.ndarray, meta: Optional[Dict[Any, Any]]
    ) -> List[Tuple[Any, np.ndarray]]:
        raise NotImplementedError()
