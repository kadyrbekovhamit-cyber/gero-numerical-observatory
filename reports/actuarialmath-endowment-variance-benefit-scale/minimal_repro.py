from pathlib import Path
import sys, json
source = Path(sys.argv[1]).resolve()
sys.path.insert(0, str(source / "src" if (source / "src").exists() else source))
import actuarialmath
from actuarialmath import LifeTable
assert Path(actuarialmath.__file__).resolve().is_relative_to(source)
life = LifeTable().set_table(q={40: .25, 41: 1}).set_interest(i=0)
args = dict(x=40, t=1, b=2, endowment=1)
print(json.dumps({"source":str(source),"mean":life.endowment_insurance(**args),"second_moment":life.endowment_insurance(**args,moment=2),"variance":life.endowment_insurance(**args,moment=life.VARIANCE),"expected_variance":3/16}))
