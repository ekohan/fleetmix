from typer.testing import CliRunner
from fleetmix.app import app

def test_exp_list():
    r = CliRunner().invoke(app, ["exp", "list"])
    assert "alpha_analysis" in r.stdout