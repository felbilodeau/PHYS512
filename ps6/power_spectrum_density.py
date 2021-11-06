import numpy as np
from windowmaker import make_flat_window

# A function to average out the psd
def get_average_psd(data, segment_length, windowing = True):
    # Since we are using rfft, the output will have segment_length / 2 + 1 elements
    noutput = int(segment_length / 2) + 1

    # This is just in case you did not want to window the data, you can set windowing = False
    # I mostly wrote this for testing purposes
    if windowing:
        window = make_flat_window(segment_length, segment_length // 6)
    else:
        window = 1

    # We calculate the number of slices given the number of data points and segment length
    # We have to make sure that segment_length is a factor of the number of data points
    # when we input, but it's not entirely required, you'll just discard some data if it isn't
    n = len(data)
    nslices = n // segment_length

    # Create the psd array
    psd = np.zeros(noutput)
    
    # For each slice, calculate the power spectrum and add it to the psd array
    for i in range(nslices):
        # Get the start and end indices given the slice
        start = i * segment_length
        end = (i + 1) * segment_length

        # Slice the data and take the FT
        data_slice = data[start:end]
        data_slice_ft = np.fft.rfft(data_slice * window)

        # Calculate and add the psd for this slice
        psd += np.abs(data_slice_ft)**2

    # Return the summed up psd
    return psd

# So this is essentially the same as the previous function except it uses np.fft.fft instead of np.fft.rfft
def get_average_psd_complex(data, segment_length, windowing = True):
    noutput = segment_length

    # This is just in case you did not want to window the data, you can set windowing = False
    # I mostly wrote this for testing purposes
    if windowing:
        window = make_flat_window(segment_length, segment_length // 6)
    else:
        window = 1

    # We calculate the number of slices given the number of data points and segment length
    # We have to make sure that segment_length is a factor of the number of data points
    # when we input, but it's not entirely required, you'll just discard some data if it isn't
    n = len(data)
    nslices = n // segment_length

    # Create the psd array
    psd = np.zeros(noutput)
    
    # For each slice, calculate the power spectrum and add it to the psd array
    for i in range(nslices):
        # Get the start and end indices given the slice
        start = i * segment_length
        end = (i + 1) * segment_length

        # Slice the data and take the FT
        data_slice = data[start:end]
        data_slice_ft = np.fft.fft(data_slice * window)

        # Calculate and add the psd for this slice
        psd += np.abs(data_slice_ft)**2

    # Return the summed up psd
    return psd