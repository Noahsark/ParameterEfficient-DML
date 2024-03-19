import datasampler.class_random_sampler
import datasampler.class_random_sampler_dist
import datasampler.random_sampler



def select(sampler, opt, image_dict, image_list=None, rank=0, **kwargs):
    if 'random' in sampler:
        sampler_lib = datasampler.class_random_sampler
    elif "dist" in sampler:
        sampler_lib = datasampler.class_random_sampler_dist
    else:
        raise Exception('Minibatch sampler <{}> not available!'.format(sampler))

    sampler = sampler_lib.Sampler(opt,image_dict=image_dict, image_list=image_list, rank=rank)
    
    return sampler
