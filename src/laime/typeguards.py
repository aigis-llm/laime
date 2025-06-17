from collections.abc import Sequence
from typing import TypeGuard


def is_str_list(val: Sequence[object], allow_empty: bool) -> TypeGuard[list[str]]:
	if len(val) == 0:
		return allow_empty
	return all(isinstance(x, str) for x in val)
