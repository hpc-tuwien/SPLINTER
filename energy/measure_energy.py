import pandas as pd
from pymeas.device import GPMDevice
from pymeas.output import CsvOutput

power_meter_device = GPMDevice(host="192.168.167.91")
power_meter_device.connect()
measurement_thread = power_meter_device.start_power_capture()
# do something here


power = power_meter_device.stop_power_capture(measurement_thread)
power_meter_device.disconnect()
data = [{'timestamp': key, 'value': value} for key, value in power.items()]
CsvOutput.save("power_measurement.csv", field_names=['timestamp', 'value'], data=data)