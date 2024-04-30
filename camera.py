import imagingcontrol4 as ic4
import time

# CHANGE WHEN USING DIFFERENT SEEDS and different runs
seed_type = "vehna"
run = "first"
path_to_conf = "settings_2604_final.json"

TIME_BETWEEN_SHOTS = 0.1
shots = 5000



def print_interface_device_tree():
    print("Enumerating video capture devices by interface...")

    interface_list = ic4.DeviceEnum.interfaces()

    if len(interface_list) == 0:
        print("No interfaces found")
        return

    for itf in interface_list:
        print(f"Interface: {itf.display_name}")
        print(f"\tProvided by {itf.transport_layer_name} [TLType: {str(itf.transport_layer_type)}]")

        device_list = itf.devices

        if len(device_list) == 0:
            print("\tNo devices found")
            continue

        print(f"\tFound {len(device_list)} devices:")

        for device_info in device_list:
            print(f"\t\t{format_device_info(device_info)}")

def format_device_info(device_info: ic4.DeviceInfo) -> str:
    return f"Model: {device_info.model_name} Serial: {device_info.serial}"

def print_device_list():
    print("Enumerating all attached video capture devices...")

    device_list = ic4.DeviceEnum.devices()

    if len(device_list) == 0:
        print("No devices found")
        return

    print(f"Found {len(device_list)} devices:")

    for device_info in device_list:
        print(format_device_info(device_info))

def main():

    # Initialize library
    ic4.Library.init()

    print("Getting connected devices...")
    print_device_list()

    # Create a Grabber object
    grabber = ic4.Grabber()

    if len(ic4.DeviceEnum.devices()) == 0:
        return
    
    #grabber.device_open(first_device_info)
    grabber.device_open_from_state_file(path_to_conf)

    # Create a SnapSink. A SnapSink allows grabbing single images (or image sequences) out of a data stream.
    sink = ic4.SnapSink()
    # Setup data stream from the video capture device to the sink and start image acquisition.
    grabber.stream_setup(sink, setup_option=ic4.StreamSetupOption.ACQUISITION_START)

    images_taken = 0
    while images_taken < shots:
        try:
            # Grab a single image out of the data stream.
            image = sink.snap_single(1000)

            # Print image information.
            print(f"Received an image. ImageType: {image.image_type}")

            # Save the image.
            path = f"photos/{seed_type}/{run}_{images_taken}.bmp"
            
            #image.save_as_png(path, compression_level=ic4.ImageBuffer.PngCompressionLevel.LOW)
            image.save_as_bmp(path)
            

        except ic4.IC4Exception as ex:
            print(ex.message)
        
        images_taken += 1
        print(f"completion:{images_taken}/{shots}")
        time.sleep(TIME_BETWEEN_SHOTS)
    
    # Stop the data stream.
    grabber.stream_stop()

        
        
        #
        # Call main
        #
print("Starting program...")
main()
