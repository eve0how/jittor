from .spec import ModelSpec
from .vm import VelocityModule
from .vm_stage1 import VelocityModuleStage1  # 改进版 - Stage1 (75分)

def get_model(model_config, **kwargs) -> ModelSpec:
    MAP = {
        'VelocityModule': VelocityModule,
        'VelocityModuleStage1': VelocityModuleStage1,  # 改进版模型
    }
    __target__ = model_config['__target__']
    del model_config['__target__']
    assert __target__ in MAP, f"expect: [{','.join(MAP.keys())}], found: {__target__}"
    return MAP[__target__](model_config=model_config, **kwargs)
