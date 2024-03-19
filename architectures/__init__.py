
import architectures.vit
import architectures.peft_vit

def select(arch, opt):
    if arch == 'vit':
        return architectures.vit.Network(opt)
    if arch == 'peft_vit':
        return architectures.peft_vit.Network(opt)
    raise NotImplementedError('Architecture not implemented.')