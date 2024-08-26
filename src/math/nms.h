#pragma once

#include <vector>
namespace cv_infer
{
void bigmeshgrid(int height, int width, float* xg, float* yg)
{
    for (int y_i = 0; y_i < height; ++y_i)
    {
        for (int x_i = 0; x_i < width; ++x_i)
        {
            xg[y_i * width + x_i] = x_i;
            yg[y_i * width + x_i] = y_i;
        }
    }
}

template <class T>
struct BigSortElement
{
    BigSortElement(){};
    BigSortElement(T v, unsigned int i) : value(v), index(i){};
    T            value;
    unsigned int index;
};

template <typename T>
struct BigDescendingSort
{
    typedef T ElementType;
    bool      operator()(const BigSortElement<T>& a, const BigSortElement<T>& b) { return a.value > b.value; }
};

std::vector<unsigned int> bigsort(std::vector<std::vector<float>>& data)
{
    // num*5
    std::vector<BigSortElement<float>> temp_vector(data.size());
    unsigned int                       index = 0;
    for (unsigned int i = 0; i < data.size(); ++i)
    {
        temp_vector[i] = BigSortElement<float>(data[i][4], i);
    }

    // sort
    BigDescendingSort<float> compare_op;
    std::sort(temp_vector.begin(), temp_vector.end(), compare_op);

    std::vector<unsigned int> result_index(data.size());
    index = 0;
    typename std::vector<BigSortElement<float>>::iterator iter, iend(temp_vector.end());
    for (iter = temp_vector.begin(); iter != iend; ++iter)
    {
        result_index[index] = ((*iter).index);
        index++;
    }

    return result_index;
}

std::vector<float> bigget_ious(std::vector<std::vector<float>>& all_bbox, std::vector<float>& target_bbox,
                               std::vector<unsigned int> order, unsigned int offset)
{
    std::vector<float> iou_list;
    for (unsigned int i = offset; i < order.size(); ++i)
    {
        int   index    = order[i];
        float inter_x1 = std::max(all_bbox[index][0], target_bbox[0]);
        float inter_y1 = std::max(all_bbox[index][1], target_bbox[1]);

        float inter_x2 = std::min(all_bbox[index][2], target_bbox[2]);
        float inter_y2 = std::min(all_bbox[index][3], target_bbox[3]);

        float inter_w = std::max(inter_x2 - inter_x1, 0.0f);
        float inter_h = std::max(inter_y2 - inter_y1, 0.0f);

        float inter_area = inter_w * inter_h;
        float a_area     = (all_bbox[index][2] - all_bbox[index][0]) * (all_bbox[index][3] - all_bbox[index][1]);
        float b_area     = (target_bbox[2] - target_bbox[0]) * (target_bbox[3] - target_bbox[1]);
        float iou        = inter_area / (a_area + b_area - inter_area);
        iou_list.push_back(iou);
    }

    return iou_list;
}

std::vector<unsigned int> bignms(std::vector<std::vector<float>>& dets, float thresh)
{
    std::vector<unsigned int> order = bigsort(dets);
    std::vector<unsigned int> keep;

    while (order.size() > 0)
    {
        unsigned int index = order[0];
        keep.push_back(index);
        if (order.size() == 1)
        {
            break;
        }

        std::vector<float>        check_ious = bigget_ious(dets, dets[index], order, 1);
        std::vector<unsigned int> remained_order;
        for (int i = 0; i < check_ious.size(); ++i)
        {
            if (check_ious[i] < thresh)
            {
                remained_order.push_back(order[i + 1]);
            }
        }
        order = remained_order;
    }
    return keep;
}
}  // namespace cv_infer