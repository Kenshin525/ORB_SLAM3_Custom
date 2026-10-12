#include "rgbd_auto_common.hpp"
#include <libobsensor/ObSensor.hpp>

using namespace rgbd_auto;

class Gemini336Source : public CameraSource {
public:
    void start() override {
        pipe_.reset(new ob::Pipeline());
        auto dev = pipe_->getDevice();
        auto info = dev->getDeviceInfo();
        cal_.backend = "OrbbecSDK_v2";
        cal_.name = info->getName();
        cal_.serial = info->getSerialNumber();
        cal_.alignment_note = "Orbbec SDK D2C depth-to-color alignment";

        auto cfg = std::make_shared<ob::Config>();
        std::shared_ptr<ob::VideoStreamProfile> color;
        try {
            auto list = pipe_->getStreamProfileList(OB_SENSOR_COLOR);
            color = list->getVideoStreamProfile(640, 480, OB_FORMAT_RGB, 30);
        } catch(...) {
            auto list = pipe_->getStreamProfileList(OB_SENSOR_COLOR);
            color = list->getVideoStreamProfile(OB_WIDTH_ANY, OB_HEIGHT_ANY, OB_FORMAT_RGB, OB_FPS_ANY);
        }
        if(!color) throw std::runtime_error("Gemini336: no RGB color profile");
        cfg->enableStream(color);

        std::shared_ptr<ob::StreamProfileList> dlist;
        OBAlignMode alignMode = ALIGN_DISABLE;
        try {
            dlist = pipe_->getD2CDepthProfileList(color, ALIGN_D2C_HW_MODE);
            if(dlist && dlist->getCount() > 0) alignMode = ALIGN_D2C_HW_MODE;
        } catch(...) {}
        if(alignMode == ALIGN_DISABLE) {
            try {
                dlist = pipe_->getD2CDepthProfileList(color, ALIGN_D2C_SW_MODE);
                if(dlist && dlist->getCount() > 0) alignMode = ALIGN_D2C_SW_MODE;
            } catch(...) {}
        }
        if(!dlist || dlist->getCount() == 0)
            dlist = pipe_->getStreamProfileList(OB_SENSOR_DEPTH);

        std::shared_ptr<ob::StreamProfile> depth;
        try {
            depth = dlist->getVideoStreamProfile(OB_WIDTH_ANY, OB_HEIGHT_ANY, OB_FORMAT_Y16, color->getFps());
        } catch(...) {
            depth = dlist->getProfile(OB_PROFILE_DEFAULT);
        }
        if(!depth) throw std::runtime_error("Gemini336: no depth profile");
        cfg->enableStream(depth);
        cfg->setAlignMode(alignMode);
        try { cfg->setFrameAggregateOutputMode(OB_FRAME_AGGREGATE_OUTPUT_ALL_TYPE_FRAME_REQUIRE); } catch(...) {}
        try { pipe_->enableFrameSync(); } catch(...) {}

        pipe_->start(cfg);

        std::shared_ptr<ob::FrameSet> fs;
        for(int i=0;i<30 && !fs;i++) fs = pipe_->waitForFrameset(200);
        if(!fs) throw std::runtime_error("Gemini336: no RGB-D frames after start");

        auto cfraw = fs->getFrame(OB_FRAME_COLOR);
        auto dfraw = fs->getFrame(OB_FRAME_DEPTH);
        if(!cfraw || !dfraw) throw std::runtime_error("Gemini336: incomplete RGB-D frameset");
        auto cf = cfraw->as<ob::ColorFrame>();
        auto df = dfraw->as<ob::DepthFrame>();

        auto cprof = cf->getStreamProfile()->as<ob::VideoStreamProfile>();
        const OBCameraParam cp = pipe_->getCameraParam();
        cal_.width  = static_cast<int>(cf->getWidth());
        cal_.height = static_cast<int>(cf->getHeight());
        cal_.fps    = static_cast<int>(cprof->getFps());
        if(static_cast<int>(cp.rgbIntrinsic.width)!=cal_.width || static_cast<int>(cp.rgbIntrinsic.height)!=cal_.height)
            throw std::runtime_error("Gemini336 STRICT: Pipeline camera parameters do not match active color resolution");
        cal_.fx=cp.rgbIntrinsic.fx; cal_.fy=cp.rgbIntrinsic.fy; cal_.cx=cp.rgbIntrinsic.cx; cal_.cy=cp.rgbIntrinsic.cy;
        cal_.dist={{cp.rgbDistortion.k1,cp.rgbDistortion.k2,cp.rgbDistortion.p1,cp.rgbDistortion.p2,cp.rgbDistortion.k3}};
        const double scale=static_cast<double>(df->getValueScale())*0.001;
        if(!(scale>0.0) || !std::isfinite(scale)) throw std::runtime_error("Gemini336 STRICT: invalid live DepthFrame value scale");
        cal_.depth_scale_m=scale;
        cal_.intrinsics_source="Orbbec Pipeline::getCameraParam() after active D2C";
        cal_.distortion_source="Orbbec Pipeline::getCameraParam().rgbDistortion";
        cal_.depth_scale_source="active Orbbec DepthFrame::getValueScale()";
        cal_.alignment_source=alignMode==ALIGN_D2C_HW_MODE ? "Orbbec SDK hardware D2C" : "Orbbec SDK software D2C";
        cal_.calibration_note="STRICT: active D2C factory/runtime calibration; no fallback intrinsics";
        cal_.strict_ok=true;

        started_ = true;
    }

    bool grab(RGBDFrame& out) override {
        if(!started_) return false;
        auto fs = pipe_->waitForFrameset(200);
        if(!fs) return false;
        auto cfraw = fs->getFrame(OB_FRAME_COLOR);
        auto dfraw = fs->getFrame(OB_FRAME_DEPTH);
        if(!cfraw || !dfraw) return false;
        auto cf = cfraw->as<ob::ColorFrame>();
        auto df = dfraw->as<ob::DepthFrame>();
        if(cf->getFormat()!=OB_FORMAT_RGB || df->getFormat()!=OB_FORMAT_Y16) return false;

        cv::Mat rgb((int)cf->getHeight(), (int)cf->getWidth(), CV_8UC3, static_cast<void*>(cf->getData()));
        cv::Mat dep((int)df->getHeight(), (int)df->getWidth(), CV_16UC1, static_cast<void*>(df->getData()));
        out.rgb = rgb.clone();
        out.depth = dep.clone();
        out.timestamp_s = (double)cf->getTimeStampUs() * 1e-6;
        out.frame_number = (uint64_t)cf->getIndex();
        return true;
    }

    void stop() override {
        if(pipe_) { try { pipe_->stop(); } catch(...) {} }
        pipe_.reset(); started_=false;
    }
    CameraCalibration calibration() const override { return cal_; }
    std::string metadata() const override { return "camera_family=Gemini336\nalign_target=color"; }
    ~Gemini336Source(){ stop(); }
private:
    std::unique_ptr<ob::Pipeline> pipe_;
    CameraCalibration cal_;
    bool started_=false;
};

int main(int argc,char** argv){
    if(argc<3 || argc>4){ std::cerr<<"Usage: "<<argv[0]<<" vocabulary template_yaml [prefix]\n"; return 1; }
    try { Gemini336Source src; return rgbd_auto::run(src,argv[1],argv[2],argc==4?argv[3]:"Gemini336"); }
    catch(const ob::Error& e){ std::cerr<<"Orbbec error: "<<e.what()<<"\n"; return 2; }
    catch(const std::exception& e){ std::cerr<<"Error: "<<e.what()<<"\n"; return 3; }
}
