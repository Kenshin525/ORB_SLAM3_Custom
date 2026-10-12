#include "rgbd_auto_common.hpp"
#include <depthai/depthai.hpp>
using namespace rgbd_auto;
class OakDProSource : public CameraSource {
public:
    void start() override {
        auto devs=dai::Device::getAllAvailableDevices();if(devs.empty())throw std::runtime_error("OAK-D Pro not detected");
        dev_.reset(new dai::Device(devs[0]));auto calib=dev_->readCalibration();
        auto K=calib.getCameraIntrinsics(dai::CameraBoardSocket::CAM_A,640,480);auto D=calib.getDistortionCoefficients(dai::CameraBoardSocket::CAM_A);auto lens=calib.getLensPosition(dai::CameraBoardSocket::CAM_A);
        if(K.size()!=3||K[0].size()!=3||K[1].size()!=3||D.size()<5)throw std::runtime_error("OAK STRICT: incomplete EEPROM RGB calibration");
        if(!(K[0][0]>0&&K[1][1]>0))throw std::runtime_error("OAK STRICT: invalid EEPROM focal length");
        if(lens==0)throw std::runtime_error("OAK STRICT: EEPROM calibrated RGB lens position unavailable");
        dai::Pipeline p;auto rgb=p.create<dai::node::ColorCamera>();auto left=p.create<dai::node::MonoCamera>();auto right=p.create<dai::node::MonoCamera>();auto stereo=p.create<dai::node::StereoDepth>();auto xoRgb=p.create<dai::node::XLinkOut>();auto xoDep=p.create<dai::node::XLinkOut>();
        xoRgb->setStreamName("rgb");xoDep->setStreamName("depth");
        rgb->setBoardSocket(dai::CameraBoardSocket::CAM_A);rgb->setResolution(dai::ColorCameraProperties::SensorResolution::THE_1080_P);rgb->setPreviewSize(640,480);rgb->setFps(30);rgb->setInterleaved(true);rgb->setColorOrder(dai::ColorCameraProperties::ColorOrder::BGR);rgb->initialControl.setManualFocus(lens);
        left->setBoardSocket(dai::CameraBoardSocket::CAM_B);right->setBoardSocket(dai::CameraBoardSocket::CAM_C);left->setResolution(dai::MonoCameraProperties::SensorResolution::THE_400_P);right->setResolution(dai::MonoCameraProperties::SensorResolution::THE_400_P);left->setFps(30);right->setFps(30);
        stereo->setDefaultProfilePreset(dai::node::StereoDepth::PresetMode::HIGH_DENSITY);stereo->setLeftRightCheck(true);stereo->setDepthAlign(dai::CameraBoardSocket::CAM_A);stereo->setOutputSize(640,480);
        rgb->preview.link(xoRgb->input);left->out.link(stereo->left);right->out.link(stereo->right);stereo->depth.link(xoDep->input);
        if(!dev_->startPipeline(p))throw std::runtime_error("OAK STRICT: failed to start calibrated pipeline");
        qrgb_=dev_->getOutputQueue("rgb",4,false);qdepth_=dev_->getOutputQueue("depth",4,false);
        cal_.backend="DepthAI_2.32_STRICT_EEPROM";cal_.name="Luxonis OAK-D Pro";cal_.serial=devs[0].getMxId();cal_.width=640;cal_.height=480;cal_.fps=30;cal_.fx=K[0][0];cal_.fy=K[1][1];cal_.cx=K[0][2];cal_.cy=K[1][2];cal_.dist={{D[0],D[1],D[2],D[3],D[4]}};cal_.depth_scale_m=0.001;
        cal_.intrinsics_source="DepthAI EEPROM getCameraIntrinsics(CAM_A,640,480)";cal_.distortion_source="DepthAI EEPROM getDistortionCoefficients(CAM_A)";cal_.depth_scale_source="DepthAI StereoDepth uint16 millimeter output";cal_.alignment_source="DepthAI StereoDepth setDepthAlign(CAM_A), LR-check enabled";cal_.alignment_note=cal_.alignment_source;cal_.calibration_note=std::string("STRICT EEPROM calibration; RGB lens fixed to EEPROM calibration position=")+std::to_string((int)lens);cal_.strict_ok=true;started_=true;
    }
    bool grab(RGBDFrame& out) override {if(!started_)return false;auto rf=qrgb_->get<dai::ImgFrame>();auto df=qdepth_->get<dai::ImgFrame>();if(!rf||!df)return false;int rw=rf->getWidth(),rh=rf->getHeight(),dw=df->getWidth(),dh=df->getHeight();auto r=rf->getData();auto d=df->getData();size_t rn=(size_t)rw*rh*3u,dn=(size_t)dw*dh*sizeof(uint16_t);if(r.size()<rn||d.size()<dn)return false;cv::Mat bgr(rh,rw,CV_8UC3,reinterpret_cast<void*>(r.data()));cv::Mat dep(dh,dw,CV_16UC1,reinterpret_cast<void*>(d.data()));cv::cvtColor(bgr,out.rgb,cv::COLOR_BGR2RGB);out.depth=dep.clone();out.timestamp_s=std::chrono::duration<double>(rf->getTimestamp().time_since_epoch()).count();out.frame_number=(uint64_t)rf->getSequenceNum();return true;}
    void stop() override {qrgb_.reset();qdepth_.reset();if(dev_){try{dev_->close();}catch(...){}}dev_.reset();started_=false;}
    CameraCalibration calibration() const override{return cal_;}
    std::string metadata() const override{return "strict_factory_calibration=1\nrgb_focus=EEPROM_calibration_lens_position\ndepth_align=CAM_A\nrgb_format=BGR888_interleaved_raw_buffer\ndepth_format=uint16_millimeter";}
    ~OakDProSource(){stop();}
private:std::unique_ptr<dai::Device> dev_;std::shared_ptr<dai::DataOutputQueue> qrgb_,qdepth_;CameraCalibration cal_;bool started_=false;
};
int main(int argc,char** argv){if(argc<3||argc>4){std::cerr<<"Usage: "<<argv[0]<<" vocabulary template_yaml [prefix]\n";return 1;}try{OakDProSource src;return rgbd_auto::run(src,argv[1],argv[2],argc==4?argv[3]:"OAK_D_Pro");}catch(const std::exception& e){std::cerr<<"OAK-D STRICT error: "<<e.what()<<"\n";return 3;}}
