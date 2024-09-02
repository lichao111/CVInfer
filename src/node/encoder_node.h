#pragma once

#include "node/node_base.h"
#include "tools/timer.h"

extern "C"
{
#include <libavcodec/avcodec.h>
#include <libavcodec/codec_id.h>
#include <libavformat/avformat.h>
#include <libavutil/avutil.h>
#include <libswscale/swscale.h>
}

namespace cv_infer
{
struct OutCfg
{
    std::string                                  out_url;
    std::string                                  codec    = "h264_nvenc";
    int                                          bit_rate = 8000000;
    int                                          width    = 1920;
    int                                          height   = 1080;
    int                                          fps      = 30;
    std::unordered_map<std::string, std::string> opt      = {{"preset", "medium"}, {"profile", "main"}, {"crf", "18"}};
};
class EncoderNode : public NodeBase
{
public:
    EncoderNode() : NodeBase(1, 0) { SetName("Decoder"); }
    EncoderNode(const std::string &file_name)
        : NodeBase(1, 0),
          OutFile(file_name){

          };
    virtual ~EncoderNode();
    bool         Init(const std::string &file_name);
    virtual bool Run() override;
    virtual bool Worker() override { return true; };

private:
    bool Open(const OutCfg &cfg);
    bool Close();
    bool PushOneFrame(const cv::Mat &frame);

    bool IsReady    = false;
    bool NeedTailer = false;
    // AVCtx
    AVFormatContext *Ctx      = nullptr;
    AVStream        *St       = nullptr;
    AVCodecContext  *CCtx     = nullptr;
    AVPacket        *Pkt      = nullptr;
    AVFrame         *Frame    = nullptr;
    SwsContext      *Sws      = nullptr;
    int              OutFlags = SWS_BILINEAR;
    std::string      OutFile;

    bool  FlushingEncodec();
    Timer CostTimer{"encoder", true};
};
}  // namespace cv_infer