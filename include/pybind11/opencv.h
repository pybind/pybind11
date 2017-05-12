#include <pybind11/pybind11.h>

#include <opencv2/core/core.hpp>


/// convert a pybind11::buffer object into a cv::Mat object
inline cv::Mat buffer2opencv ( pybind11::buffer &b )
{
        const pybind11::buffer_info info = b.request();

        int opencv_type=-1;
        if ( info.format==pybind11::format_descriptor<double>::value ) {
                opencv_type=CV_64F;
        }
        if ( info.format==pybind11::format_descriptor<float>::value ) {
                opencv_type=CV_32F;
        }
        if ( info.format==pybind11::format_descriptor<unsigned char>::value ) {
                opencv_type=CV_8U;
        }

        if ( opencv_type<0 || info.ndim > 8 )
                throw std::runtime_error ( "Incompatible buffer format!" );

        int ndims = info.ndim;

        int sizes[ndims];
        size_t steps[ndims];

        for ( size_t i =0; i<info.ndim; i++ ) {
                sizes[i] = info.shape[i];
	}
        for ( size_t i =0; i<info.ndim-1; i++ ) {
                steps[i] = info.strides[i];
        }
        cv::Mat m ( ndims, sizes, opencv_type, info.ptr, steps );
        return m.clone(); // just to be safe
}
