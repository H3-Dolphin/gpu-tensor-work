#include "RealtimeO1CPU.hpp"
#include "utils.hpp"

using namespace cv;
using namespace std;

// まず0,stride,stride*2, ... , 256の(1<<bit_img)/stride + 1通りで作成
// 1. Wk(y),Jk(y)を作る。Matを(1<<bit_img)/stride + 1枚ずつ
// 2. (2r+1)*(2r+1)のGFとのアダマール積の総和を計算して、出力を分子、分母Mat (1<<bit_img)/stride + 1枚ずつに格納する
// 3. 割り算して例の中間生成物を作る
// 4. srcをラスタ走査して、近い二つの中間生成物から、線形補間してdestを得る
void RealtimeO1Scala(const cv::Mat& src, cv::Mat& dest, const int r, const double sigma_r, const double sigma_s)
{
	CV_Assert(src.channels() == 1);
	const int stride = 32;
	const int bit_img = 8;
	const int num_Mats = (1<<bit_img)/stride + 1;
	// パディングした入力tempと、出力dest
	Mat temp;
	copyMakeBorder(src, temp, r, r, r, r, BORDER_REPLICATE);
	dest = Mat(src.rows, src.cols, CV_64FC1);

	// num_Mats枚の、WkとJk、2でできる分子、分母
	vector<Mat> Wk, Jk,numerator,denominator;
	Wk.resize(num_Mats);
	Jk.resize(num_Mats);
	numerator.resize(num_Mats);
	denominator.resize(num_Mats);

	// サイズと型設定
	for (int i = 0; i < num_Mats; i++)
	{
		Wk[i] = Mat(temp.rows, temp.cols, CV_64FC1);
		Jk[i] = Mat(temp.rows, temp.cols, CV_64FC1);
		numerator[i] = Mat(src.rows, src.cols, CV_64FC1);
		denominator[i] = Mat(src.rows, src.cols, CV_64FC1);
	}

	// 1. num_Mats枚のJkとWkを作成
	for (int i = 0; i < num_Mats; i++)
	{
		for (int j = 0; j < temp.rows; j++)
		{
			auto temp_ptr = temp.ptr<double>(j);
			auto Wki_ptr = Wk[i].ptr<double>(j);
			auto Jki_ptr = Jk[i].ptr<double>(j);

			for (int k = 0; k < temp.cols; k++)
			{
				const auto temp_ptrx = temp_ptr + k;
				auto Wki_ptrx = Wki_ptr + k;
				auto Jki_ptrx = Jki_ptr + k;
				double tmp;
				tmp = exp(-(i*stride-(*temp_ptrx))* (i * stride - (*temp_ptrx))/(2.0*sigma_r*sigma_r));
				*Wki_ptrx = tmp;
				*Jki_ptrx = tmp* (*temp_ptrx);
			}
		}
	}

	//double t = exp(-(30 * 30) / (2.f * sigma_r * sigma_r));
	//Mat test = Wk[5];
	//Mat test2 = Jk[5];
	
	// 2. GFとのアダマール積の総和
	vector<double> weights((2 * r + 1) * (2 * r + 1));
	vector<int> space_ofs((2 * r + 1) * (2 * r + 1));
	int maxk;
	createGaussianKernel(&weights[0], &space_ofs[0], maxk, r, sigma_r, temp.cols, true);


	for (int i = 0; i < num_Mats; i++)
	{
		#pragma omp parallel for
		for (int y = 0; y < src.rows; ++y)
		{
			// WとJのptr用意
			auto Wki_ptr = Wk[i].ptr<double>(y + r) + r;
			auto Jki_ptr = Jk[i].ptr<double>(y + r) + r;
			auto numerator_ptr = numerator[i].ptr<double>(y);
			auto denominator_ptr = denominator[i].ptr<double>(y);

			for (int x = 0; x < src.cols; ++x)
			{
				double sum_W = 0.f;
				double wsum_W = 0.f;
				double sum_J = 0.f;
				double wsum_J = 0.f;

				const auto Wki_ptrx = Wki_ptr + x;
				const auto Jki_ptrx = Jki_ptr + x;
				auto numerator_ptrx = numerator_ptr + x;
				auto denominator_ptrx = denominator_ptr + x;

				// ここでアダマール積の総和
				for (int k = 0; k < maxk; ++k)
				{
					// Wk
					sum_W += double(*(Wki_ptrx + space_ofs[k])) * weights[k];
					wsum_W += weights[k];

					// Jk
					sum_J += double(*(Jki_ptrx + space_ofs[k])) * weights[k];
					wsum_J += weights[k];
				}
				// 分子にJkの結果、分母にWkの結果を入れる
				*numerator_ptrx = (sum_J / wsum_J);
				*denominator_ptrx = (sum_W / wsum_W);
			}
		}
	}

	
	// 3. 割り算で、中間生成物作成(PBFIC)作る、numeratorを使いまわせばOK
	for (int i = 0; i < num_Mats; i++)
	{
		max(0.0005, denominator[i],denominator[i]);
		divide(numerator[i],denominator[i],numerator[i]);
	}

	// 4. ラスタ走査して、0,stride,stride*2, ... ,256の間の数を
	// PBFIC(実装上はnumerator)から線形補間
	for(int y = 0;y < src.rows;y++)
	{
		auto src_ptr = src.ptr<double>(y);
		auto dest_ptr = dest.ptr<double>(y);

		for (int x = 0; x < src.cols; x++)
		{
			const auto src_ptrx = src_ptr + x;
			auto dest_ptrx = dest_ptr + x;
			for(int i = 0;i < (1<<bit_img);i+=stride)
			{
				if(*src_ptr >= i && (*src_ptr < i+stride))
				{
					// 1個次と補間
					auto numeratori_ptr = numerator[i/stride].ptr<double>(y);
					auto numeratori_ptrx = numeratori_ptr+ x;
					auto numeratorinext_ptr = numerator[i/stride+1].ptr<double>(y);
					auto numeratorinext_ptrx = numeratorinext_ptr + x;
					// 線形補間
					*dest_ptrx = ((i+stride) - *src_ptrx) / (double)stride * (*numeratori_ptrx) + (*src_ptrx - i) / (double)stride * (*numeratorinext_ptrx);
				}
			}
		}
	}
}