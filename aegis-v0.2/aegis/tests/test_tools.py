from core.tools import RISK_HIGH, RISK_LOW, TOOLS


def test_registry_has_expected_tools():
    assert set(TOOLS) == {"list_files", "read_file", "system_info", "run_command"}


def test_run_command_is_high_risk():
    assert TOOLS["run_command"].risk_category == RISK_HIGH


def test_read_only_tools_are_low_risk():
    assert TOOLS["list_files"].risk_category == RISK_LOW
    assert TOOLS["read_file"].risk_category == RISK_LOW
    assert TOOLS["system_info"].risk_category == RISK_LOW


def test_list_files_lists_a_real_directory(tmp_path):
    (tmp_path / "a.txt").write_text("hi")
    (tmp_path / "sub").mkdir()
    result = TOOLS["list_files"].handler(str(tmp_path))
    assert "a.txt" in result
    assert "sub/" in result


def test_read_file_reads_contents(tmp_path):
    f = tmp_path / "note.txt"
    f.write_text("hello world")
    result = TOOLS["read_file"].handler(str(f))
    assert result == "hello world"


def test_read_file_missing_path_reports_error():
    result = TOOLS["read_file"].handler("/definitely/not/a/real/path.txt")
    assert result.startswith("Error")


def test_system_info_returns_something_plausible():
    result = TOOLS["system_info"].handler()
    assert "OS:" in result
    assert "Python:" in result
