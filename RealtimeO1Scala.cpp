#include "RealtimeO1CPU.hpp"
#include "utils.hpp"

using namespace cv;
using namespace std;

// �܂�0,stride,stride*2, ... , 256��(1<<bit_img)/stride + 1�ʂ�ō쐬
// 1. Wk(y),Jk(y)�����BMat��(1<<bit_img)/stride + 1������
// 2. (2r+1)*(2r+1)��GF�Ƃ̃A�_�}�[���ς̑��a���v�Z���āA�o�͂𕪎q�A����Mat (1<<bit_img)/stride + 1�����Ɋi�[����
// 3. ����Z���ė�̒��Ԑ����������
// 4. src�����X�^�������āA�߂���̒��Ԑ���������A���`��Ԃ���dest�𓾂�
void RealtimeO1Scala(const cv::Mat& src, cv::Mat& dest, const int r, const double sigma_r, const double sigma_s)
{
	CV_Assert(src.channels() == 1);
	const int stride = 32;
	const int bit_img = 8;
	const int num_Mats = (1<<bit_img)/stride + 1;
	// �p�f�B���O��������temp�ƁA�o��dest
	Mat temp;
	copyMakeBorder(src, temp, r, r, r, r, BORDER_REPLICATE);
	dest = Mat(src.rows, src.cols, CV_64FC1);

	// num_Mats���́AWk��Jk�A2�łł��镪�q�A����
	vector<Mat> Wk, Jk,numerator,denominator;
	Wk.resize(num_Mats);
	Jk.resize(num_Mats);
	numerator.resize(num_Mats);
	denominator.resize(num_Mats);

	// �T�C�Y�ƌ^�ݒ�
	for (int i = 0; i < num_Mats; i++)
	{
		Wk[i] = Mat(temp.rows, temp.cols, CV_64FC1);
		Jk[i] = Mat(temp.rows, temp.cols, CV_64FC1);
		numerator[i] = Mat(src.rows, src.cols, CV_64FC1);
		denominator[i] = Mat(src.rows, src.cols, CV_64FC1);
	}

	// 1. num_Mats����Jk��Wk���쐬
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
	
	// 2. GF�Ƃ̃A�_�}�[���ς̑��a
	vector<double> weights((2 * r + 1) * (2 * r + 1));
	vector<int> space_ofs((2 * r + 1) * (2 * r + 1));
	int maxk;
	createGaussianKernel(&weights[0], &space_ofs[0], maxk, r, sigma_r, temp.cols, true);


	for (int i = 0; i < num_Mats; i++)
	{
		#pragma omp parallel for
		for (int y = 0; y < src.rows; ++y)
		{
			// W��J��ptr�p��
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

				// �����ŃA�_�}�[���ς̑��a
				for (int k = 0; k < maxk; ++k)
				{
					// Wk
					sum_W += double(*(Wki_ptrx + space_ofs[k])) * weights[k];
					wsum_W += weights[k];

					// Jk
					sum_J += double(*(Jki_ptrx + space_ofs[k])) * weights[k];
					wsum_J += weights[k];
				}
				// ���q��Jk�̌��ʁA�����Wk�̌��ʂ�����
				*numerator_ptrx = (sum_J / wsum_J);
				*denominator_ptrx = (sum_W / wsum_W);
			}
		}
	}

	
	// 3. ����Z�ŁA���Ԑ������쐬(PBFIC)���Anumerator���g���܂킹��OK
	for (int i = 0; i < num_Mats; i++)
	{
		max(0.0005, denominator[i],denominator[i]);
		divide(numerator[i],denominator[i],numerator[i]);
	}

	// 4. ���X�^�������āA0,stride,stride*2, ... ,256�̊Ԃ̐���
	// PBFIC(�������numerator)������`���
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
					// 1���ƕ��
					auto numeratori_ptr = numerator[i/stride].ptr<double>(y);
					auto numeratori_ptrx = numeratori_ptr+ x;
					auto numeratorinext_ptr = numerator[i/stride+1].ptr<double>(y);
					auto numeratorinext_ptrx = numeratorinext_ptr + x;
					// ���`���
					*dest_ptrx = ((i+stride) - *src_ptrx) / (double)stride * (*numeratori_ptrx) + (*src_ptrx - i) / (double)stride * (*numeratorinext_ptrx);
				}
			}
		}
	}
}