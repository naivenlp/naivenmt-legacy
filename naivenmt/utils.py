def get_device_str(device_id, num_gpus):
  if num_gpus == 0:
    return "/cpu:0"
  device_str = "/gpu:%d" % (device_id % num_gpus)
  return device_str
