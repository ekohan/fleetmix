import sys
import pytest
from fleetmix.cli.vrp_to_fsm import main as unified_main


def test_unified_info_flag(capsys):
    sys.argv = ['prog', '--info']
    unified_main()
    out, err = capsys.readouterr()
    assert 'Unified VRP-to-FSM Conversion Tool' in out 