from laime.typeguards import is_str_list


def test_is_str_list():
	assert is_str_list(["a", "b"], False)


def test_is_str_list_not():
	assert not is_str_list([1, 2], False)


def test_is_str_list_empty():
	assert is_str_list([], True)
