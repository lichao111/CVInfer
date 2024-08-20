#pragma once

#include <string>
#include <unordered_map>

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
class DecoderNode : public NodeBase
{
public:
    DecoderNode() : NodeBase(0, 1, 1){};
    DecoderNode(const std::string& source)
        : NodeBase(0, 1, 1),
          URI(source){

          };
    virtual ~DecoderNode() = default;
    bool                               Init(const std::string& source);
    virtual bool                       Run() override;
    virtual std::vector<SignalBasePtr> Worker(
        std::vector<SignalBasePtr> input_signals) override
    {
        return {};
    };

private:
    bool                     Open();
    bool                     Close();
    cv::Mat                  GetOneFrame();
    AVCodecContext*          GetAVCodecContext(int idx) const;
    int                      GetFirstStreamByType(enum AVMediaType type) const;
    std::vector<std::string> GetDecoderNameByCodecId(
        const AVCodecID codec_id) const;
    cv::Mat DecodeToFrame(AVFrame* frame);
    cv::Mat TryFlushing();

private:
    std::string URI;

    AVFormatContext* Ctx   = nullptr;
    AVCodecContext*  CCtx  = nullptr;
    AVPacket*        Pkt   = nullptr;
    AVFrame*         Frame = nullptr;
    SwsContext*      Sws   = nullptr;

    int           OutFlags   = SWS_BILINEAR;
    int           StreamIdx  = 0;
    int           Width      = 0;
    int           Height     = 0;
    int           ThreadNum  = 1;
    std::uint64_t FrameIndex = 0;

    bool NeedFlushing = false;
    bool VideoEOF     = false;

    std::unordered_map<AVCodecID, std::vector<std::string>> DecodersPriority = {
        {AV_CODEC_ID_H264, {"h264"}}};

    Timer CostTimer{"decoder"};
};
}  // namespace cv_infer