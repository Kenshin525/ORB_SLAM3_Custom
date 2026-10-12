#include "rgbd_auto_common.hpp"
#include <OpenNI.h>
#include <PS1080.h>
#include <array>
#include <cstring>
#include <cmath>

using namespace rgbd_auto;

class AstraSource : public CameraSource {
public:
    void start() override {
        if(openni::OpenNI::initialize() != openni::STATUS_OK) {
            throw std::runtime_error(
                std::string("OpenNI init failed: ") +
                openni::OpenNI::getExtendedError()
            );
        }

        if(dev_.open(openni::ANY_DEVICE) != openni::STATUS_OK) {
            throw std::runtime_error(
                std::string("OpenNI device open failed: ") +
                openni::OpenNI::getExtendedError()
            );
        }

        const auto info = dev_.getDeviceInfo();

        cal_.backend = "OpenNI2_STRICT_FACTORY_LEGACY";
        cal_.name = info.getName() ? info.getName() : "Orbbec Astra";
        cal_.name += " (IR+Depth)";
        cal_.serial = info.getUri() ? info.getUri() : "unknown";

        if(depth_.create(dev_, openni::SENSOR_DEPTH) != openni::STATUS_OK) {
            throw std::runtime_error(
                std::string("Astra STRICT: depth stream create failed: ") +
                openni::OpenNI::getExtendedError()
            );
        }

        if(ir_.create(dev_, openni::SENSOR_IR) != openni::STATUS_OK) {
            throw std::runtime_error(
                std::string("Astra STRICT: IR stream create failed: ") +
                openni::OpenNI::getExtendedError()
            );
        }

        choose_strict_mode();

        // This installed Orbbec OpenNI2 SDK exposes the legacy factory
        // calibration structure OBCameraParams:
        //   l_intr_p[4] = [fx,fy,cx,cy]
        //   r_intr_p[4] = [fx,fy,cx,cy]
        //   l_k/r_k     = [k1,k2,p1,p2,k3]
        //
        // On the legacy Astra optical layout, the left calibration is the
        // depth/IR optical channel used here. The external UVC RGB camera is
        // intentionally not used.
        OBCameraParams factory{};
        int dataSize = static_cast<int>(sizeof(factory));

        const openni::Status rc = dev_.getProperty(
            openni::OBEXTENSION_ID_CAM_PARAMS,
            reinterpret_cast<uint8_t*>(&factory),
            &dataSize
        );

        if(rc != openni::STATUS_OK) {
            throw std::runtime_error(
                std::string(
                    "Astra STRICT: OBEXTENSION_ID_CAM_PARAMS unavailable: "
                ) + openni::OpenNI::getExtendedError()
            );
        }

        if(dataSize < static_cast<int>(sizeof(factory))) {
            throw std::runtime_error(
                "Astra STRICT: factory calibration payload is incomplete"
            );
        }

        const auto mode = ir_.getVideoMode();
        const int w = mode.getResolutionX();
        const int h = mode.getResolutionY();

        // Strict mode intentionally runs only at the legacy factory calibration
        // resolution. No scaling or guessed principal point is allowed.
        if(w != 640 || h != 480) {
            throw std::runtime_error(
                "Astra STRICT: active IR/depth mode is not 640x480; "
                "automatic calibration scaling is disabled"
            );
        }

        const double fx0 = factory.l_intr_p[0];
        const double fy0 = factory.l_intr_p[1];
        const double cx0 = factory.l_intr_p[2];
        const double cy0 = factory.l_intr_p[3];

        if(!std::isfinite(fx0) || !std::isfinite(fy0) ||
           !std::isfinite(cx0) || !std::isfinite(cy0) ||
           fx0 <= 0.0 || fy0 <= 0.0 ||
           cx0 < 0.0 || cx0 >= w ||
           cy0 < 0.0 || cy0 >= h) {
            throw std::runtime_error(
                "Astra STRICT: invalid factory depth/IR intrinsics"
            );
        }

        for(int i = 0; i < 5; ++i) {
            if(!std::isfinite(factory.l_k[i])) {
                throw std::runtime_error(
                    "Astra STRICT: invalid factory depth/IR distortion"
                );
            }
        }

        if(depth_.start() != openni::STATUS_OK) {
            throw std::runtime_error(
                std::string("Astra STRICT: depth stream start failed: ") +
                openni::OpenNI::getExtendedError()
            );
        }

        if(ir_.start() != openni::STATUS_OK) {
            depth_.stop();
            throw std::runtime_error(
                std::string("Astra STRICT: IR stream start failed: ") +
                openni::OpenNI::getExtendedError()
            );
        }

        openni::VideoFrameRef irf, df;
        for(int i = 0; i < 60; ++i) {
            const auto ds = depth_.readFrame(&df);
            const auto is = ir_.readFrame(&irf);
            if(ds == openni::STATUS_OK &&
               is == openni::STATUS_OK &&
               df.isValid() &&
               irf.isValid()) {
                break;
            }
        }

        if(!df.isValid() || !irf.isValid()) {
            throw std::runtime_error(
                "Astra STRICT: no valid IR+Depth frames"
            );
        }

        if(df.getVideoMode().getPixelFormat() !=
           openni::PIXEL_FORMAT_DEPTH_1_MM) {
            throw std::runtime_error(
                "Astra STRICT: active depth format is not DEPTH_1_MM"
            );
        }

        if(df.getWidth() != w || df.getHeight() != h) {
            throw std::runtime_error(
                "Astra STRICT: IR/depth resolutions differ"
            );
        }

        cal_.width = w;
        cal_.height = h;
        cal_.fps = mode.getFps();

        // This physical unit's OpenNI IR/depth image needs a horizontal
        // orientation correction. Because both images are flipped together,
        // transform the real factory pinhole/distortion parameters exactly.
        cal_.fx = fx0;
        cal_.fy = fy0;
        cal_.cx = static_cast<double>(w - 1) - cx0;
        cal_.cy = cy0;

        // For horizontal image reflection:
        // k1,k2,k3 unchanged; p1 unchanged; p2 changes sign.
        cal_.dist = std::array<double,5>{{
            static_cast<double>(factory.l_k[0]),
            static_cast<double>(factory.l_k[1]),
            static_cast<double>(factory.l_k[2]),
            -static_cast<double>(factory.l_k[3]),
            static_cast<double>(factory.l_k[4])
        }};

        cal_.depth_scale_m = 0.001;

        cal_.intrinsics_source =
            "device OBEXTENSION_ID_CAM_PARAMS OBCameraParams.l_intr_p";
        cal_.distortion_source =
            "device OBEXTENSION_ID_CAM_PARAMS OBCameraParams.l_k";
        cal_.depth_scale_source =
            "active OpenNI PIXEL_FORMAT_DEPTH_1_MM";
        cal_.alignment_source =
            "native Astra IR/depth optical channel; same horizontal transform applied to both";

        cal_.alignment_note = cal_.alignment_source;
        cal_.calibration_note =
            "STRICT legacy factory calibration; no FOV-derived focal length, "
            "no image-center fallback, no guessed depth scale";

        cal_.strict_ok = true;
        ir_format_ = irf.getVideoMode().getPixelFormat();
        started_ = true;

        std::cout << "\nAstra factory calibration payload:\n"
                  << "  left/depth-IR K : "
                  << factory.l_intr_p[0] << ", "
                  << factory.l_intr_p[1] << ", "
                  << factory.l_intr_p[2] << ", "
                  << factory.l_intr_p[3] << "\n"
                  << "  right/UVC K     : "
                  << factory.r_intr_p[0] << ", "
                  << factory.r_intr_p[1] << ", "
                  << factory.r_intr_p[2] << ", "
                  << factory.r_intr_p[3] << "\n"
                  << "  left distortion: "
                  << factory.l_k[0] << ", "
                  << factory.l_k[1] << ", "
                  << factory.l_k[2] << ", "
                  << factory.l_k[3] << ", "
                  << factory.l_k[4] << "\n"
                  << "  factory r2l t  : "
                  << factory.r2l_t[0] << ", "
                  << factory.r2l_t[1] << ", "
                  << factory.r2l_t[2] << "\n"
                  << "  active mode    : "
                  << w << "x" << h << "@" << cal_.fps << "\n";
    }

    bool grab(RGBDFrame& out) override {
        if(!started_) return false;

        openni::VideoFrameRef irf, df;

        if(depth_.readFrame(&df) != openni::STATUS_OK) return false;
        if(ir_.readFrame(&irf) != openni::STATUS_OK) return false;
        if(!df.isValid() || !irf.isValid()) return false;

        cv::Mat gray;

        const auto pf = irf.getVideoMode().getPixelFormat();

        if(pf == openni::PIXEL_FORMAT_GRAY8) {
            cv::Mat g(
                irf.getHeight(),
                irf.getWidth(),
                CV_8UC1,
                const_cast<void*>(irf.getData())
            );
            gray = g.clone();
        }
        else if(pf == openni::PIXEL_FORMAT_GRAY16) {
            cv::Mat g16(
                irf.getHeight(),
                irf.getWidth(),
                CV_16UC1,
                const_cast<void*>(irf.getData())
            );

            // Fixed conversion only; no frame-dependent min/max normalization.
            g16.convertTo(gray, CV_8U, 1.0 / 256.0);
        }
        else {
            return false;
        }

        cv::Mat grayFlip;
        cv::flip(gray, grayFlip, 1);
        cv::cvtColor(grayFlip, out.rgb, cv::COLOR_GRAY2RGB);

        cv::Mat dep(
            df.getHeight(),
            df.getWidth(),
            CV_16UC1,
            const_cast<void*>(df.getData())
        );

        cv::flip(dep, out.depth, 1);

        out.timestamp_s =
            static_cast<double>(irf.getTimestamp()) * 1e-6;

        out.frame_number =
            static_cast<uint64_t>(irf.getFrameIndex());

        return true;
    }

    void stop() override {
        if(started_) {
            try { ir_.stop(); } catch(...) {}
            try { depth_.stop(); } catch(...) {}
        }

        try { ir_.destroy(); } catch(...) {}
        try { depth_.destroy(); } catch(...) {}
        try { dev_.close(); } catch(...) {}

        openni::OpenNI::shutdown();
        started_ = false;
    }

    CameraCalibration calibration() const override {
        return cal_;
    }

    std::string metadata() const override {
        return
            "strict_factory_calibration=1\n"
            "astra_factory_struct=OBCameraParams\n"
            "astra_visual_stream=IR\n"
            "astra_uvc_color_used=0\n"
            "astra_horizontal_flip=1\n"
            "astra_ir_mapping=GRAY16_to_GRAY8_fixed_div256\n"
            "fallback_sensor_calibration=DISABLED";
    }

    ~AstraSource() {
        stop();
    }

private:
    void choose_strict_mode() {
        const auto& ds =
            depth_.getSensorInfo().getSupportedVideoModes();

        const auto& is =
            ir_.getSensorInfo().getSupportedVideoModes();

        bool found = false;
        openni::VideoMode dm, im;

        // Strict legacy calibration uses the factory 640x480 model only.
        // Prefer 30 fps; allow 15 fps without changing geometry.
        const int fpsPref[] = {30, 15};

        for(int fps : fpsPref) {
            for(int di = 0; di < ds.getSize() && !found; ++di) {
                const auto& d = ds[di];

                if(d.getPixelFormat() !=
                   openni::PIXEL_FORMAT_DEPTH_1_MM) continue;

                if(d.getResolutionX() != 640 ||
                   d.getResolutionY() != 480 ||
                   d.getFps() != fps) continue;

                for(int ii = 0; ii < is.getSize(); ++ii) {
                    const auto& ir = is[ii];

                    const bool goodFormat =
                        ir.getPixelFormat() ==
                            openni::PIXEL_FORMAT_GRAY16 ||
                        ir.getPixelFormat() ==
                            openni::PIXEL_FORMAT_GRAY8;

                    if(goodFormat &&
                       ir.getResolutionX() == 640 &&
                       ir.getResolutionY() == 480 &&
                       ir.getFps() == fps) {
                        dm = d;
                        im = ir;
                        found = true;
                        break;
                    }
                }
            }

            if(found) break;
        }

        if(!found) {
            throw std::runtime_error(
                "Astra STRICT: exact 640x480 common IR + DEPTH_1_MM mode unavailable"
            );
        }

        if(depth_.setVideoMode(dm) != openni::STATUS_OK) {
            throw std::runtime_error(
                "Astra STRICT: failed to set real depth mode"
            );
        }

        if(ir_.setVideoMode(im) != openni::STATUS_OK) {
            throw std::runtime_error(
                "Astra STRICT: failed to set real IR mode"
            );
        }
    }

    openni::Device dev_;
    openni::VideoStream depth_;
    openni::VideoStream ir_;

    CameraCalibration cal_;

    bool started_ = false;
    openni::PixelFormat ir_format_ =
        openni::PIXEL_FORMAT_GRAY16;
};

int main(int argc, char** argv) {
    if(argc < 3 || argc > 4) {
        std::cerr
            << "Usage: " << argv[0]
            << " vocabulary template_yaml [prefix]\n";
        return 1;
    }

    try {
        AstraSource src;

        return rgbd_auto::run(
            src,
            argv[1],
            argv[2],
            argc == 4 ? argv[3] : "Astra"
        );
    }
    catch(const std::exception& e) {
        std::cerr
            << "Astra/OpenNI STRICT error: "
            << e.what()
            << "\n";
        return 3;
    }
}
