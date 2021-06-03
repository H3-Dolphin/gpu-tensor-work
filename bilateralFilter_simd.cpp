#include "bilateralFilter_simd.hpp"
using namespace cv;
using namespace std;

void bilateralTest()
{
	int sigma_s_ = 10;
	int sigma_r_ = 160;
	const string wname = "dest";
	namedWindow("dest");
	createTrackbar("sigma_s", wname, &sigma_s_, 100);
	createTrackbar("sigma_r", wname, &sigma_r_, 1000);
	Mat src = cv::imread("img/lenna.png");


	int key = 0;
	vector<double> calcTimes;
	while (key != 'q')
	{
		const float sigma_s = sigma_s_ / 10.f;
		const float sigma_r = sigma_r_ / 10.f;
		const int r = (int)(3.f * sigma_s);

		Mat dest;
		Mat dest_simd;
		Mat dest_simd_kernel;
		TickMeter tmeter;
		{
			tmeter.start();
			BilateralFilter(src, dest, r, sigma_r, sigma_s);
			//BilateralFilter_SIMD(src, dest_simd, r, sigma_r, sigma_s);
			BilateralFilter_SIMD_kernelloop(src, dest_simd_kernel, r, sigma_r, sigma_s);
			tmeter.stop();
		}
		calcTimes.push_back(tmeter.getTimeMilli());
		std::sort(calcTimes.begin(), calcTimes.end());
		std::cout << "time (median): " << calcTimes[calcTimes.size() / 2] << " ms" << std::endl;
		//std::cout << "PSNR : " << PSNR(dest, dest_simd) << " dB" << std::endl;
		std::cout << "PSNR : " << PSNR(dest, dest_simd_kernel) << " dB" << std::endl;

		imshow("src", src);
		imshow("dest", dest);
		//imshow("dest_simd", dest_simd);
		//imshow("dest_simd_kernel", dest_simd_kernel);
		//imwrite("bilateral_scala.png", dest);
		//imwrite("bilateral_simd.png", dest_simd);
		//imwrite("bilateral_simd_kernelloop.png", dest_simd_kernel);
		key = waitKey(5);
	}
}


void inline _mm256_stream_ps_color(void* dst, const __m256 rsrc, const __m256 gsrc, const __m256 bsrc)
{
	const int smask1 = _MM_SHUFFLE(1, 2, 3, 0);
	const int smask2 = _MM_SHUFFLE(2, 3, 0, 1);
	const int smask3 = _MM_SHUFFLE(3, 0, 1, 2);
	const int bmask1 = 0x44;
	const int bmask2 = 0x22;
	const int pmask1 = 0x20;
	const int pmask2 = 0x30;
	const int pmask3 = 0x31;
	const __m256 aa = _mm256_shuffle_ps(rsrc, rsrc, smask1);
	const __m256 bb = _mm256_shuffle_ps(gsrc, gsrc, smask2);
	const __m256 cc = _mm256_shuffle_ps(bsrc, bsrc, smask3);
	__m256 bval = _mm256_blend_ps(_mm256_blend_ps(aa, cc, bmask1), bb, bmask2);
	__m256 gval = _mm256_blend_ps(_mm256_blend_ps(cc, bb, bmask1), aa, bmask2);
	__m256 rval = _mm256_blend_ps(_mm256_blend_ps(bb, aa, bmask1), cc, bmask2);
	_mm256_stream_ps((float*)dst + 0, _mm256_permute2f128_ps(bval, rval, pmask1));
	_mm256_stream_ps((float*)dst + 8, _mm256_permute2f128_ps(gval, bval, pmask2));
	_mm256_stream_ps((float*)dst + 16, _mm256_permute2f128_ps(rval, gval, pmask3));
}

float* createGaussianKernel(const int radius, float sigma)
{
	float* kernel = (float*)_mm_malloc(sizeof(float) * (2 * radius + 1) * (2 * radius + 1), 32);

	int idx = 0;
	float sum = 0.f;
	for (int i = -radius; i <= radius; i++)
	{
		for (int j = -radius; j <= radius; j++)
		{
			const float v = exp((i * i + j * j) / (-2.f * sigma * sigma));
			kernel[idx++] = v;
			sum += v;
		}
	}

	idx = 0;
	sum = 1.f / sum;
	for (int i = -radius; i <= radius; i++)
	{
		for (int j = -radius; j <= radius; j++)
		{
			kernel[idx++] *= sum;
		}
	}

	return kernel;
}

// �o�C���e�����t�B���^�A�X�J�������i�J���[�j
void BilateralFilter(const Mat& src, Mat& dest, const int r, const float sigma_r, const float sigma_s)
{
	Mat temp_8u;
	copyMakeBorder(src, temp_8u, r, r, r, r, BORDER_REPLICATE);

	// �o��
	Mat dest_32f(src.rows, src.cols, CV_32FC3);
	Mat splitdest[3];
	split(src, splitdest);

	if (src.channels() == 3)
	{
		Mat splitImg[3];
		split(temp_8u, splitImg);

		for (int y = 0; y < src.rows; y++)
		{
			// ����|�C���^�ĂԂ̒x������A�����Ő錾����++
			uchar* tgt_r = splitImg[0].ptr<uchar>(y + r);
			uchar* tgt_g = splitImg[1].ptr<uchar>(y + r);
			uchar* tgt_b = splitImg[2].ptr<uchar>(y + r);

			// �ŏI���ʊi�[�p
			uchar* kakunou_r = splitdest[0].ptr<uchar>(y);
			uchar* kakunou_g = splitdest[1].ptr<uchar>(y);
			uchar* kakunou_b = splitdest[2].ptr<uchar>(y);

			for (int x = 0; x < src.cols; x++)
			{
				float sum[3] = { 0 };
				float wsum = 0;

				for (int i = -r; i <= r; i++)
				{

					// data�̓_��
					uchar* ref_r = splitImg[0].ptr<uchar>(y + r + i);
					uchar* ref_g = splitImg[1].ptr<uchar>(y + r + i);
					uchar* ref_b = splitImg[2].ptr<uchar>(y + r + i);

					// ����++���Ă����Ēl�����킹�邽�߂�r����
					ref_r -= r;
					ref_g -= r;
					ref_b -= r;

					for (int j = -r; j <= r; j++)
					{
						const int space_distance = j * j + i * i;
						const float ws = exp(-space_distance / (2.f * sigma_s * sigma_s));

						const float range_distance = (ref_r[x + r] - tgt_r[r]) * (ref_r[x + r] - tgt_r[r]) + (ref_g[x + r] - tgt_g[r]) * (ref_g[x + r] - tgt_g[r]) + (ref_b[x + r] - tgt_b[r]) * (ref_b[x + r] - tgt_b[r]);
						const float wr = exp(-range_distance / (2.f * sigma_r * sigma_r));
						const float w = ws * wr;

						sum[0] += ref_r[x + r] * w;
						sum[1] += ref_g[x + r] * w;
						sum[2] += ref_b[x + r] * w;

						wsum += w;

						ref_r++;
						ref_g++;
						ref_b++;
					}
				}

				kakunou_r[x] = sum[0] / wsum + 0.5f;
				kakunou_g[x] = sum[1] / wsum + 0.5f;
				kakunou_b[x] = sum[2] / wsum + 0.5f;

				tgt_r++;
				tgt_g++;
				tgt_b++;
			}
		}
	}
	else if (src.channels() == 1)
	{
		//gray image
		std::cout << "not support gray image" << std::endl;
		return;
	}
	// �������ċ��߂�splitdest��merge����
	merge(splitdest, 3, dest_32f);

	dest = Mat(dest_32f);
}

// SIMD����(��f���[�v�W�J)
void BilateralFilter_SIMD(const Mat& src, Mat& dest, const int r, float sigma_r, float sigma_s)
{
	//generate Gaussian kernel
	float* kernel = createGaussianKernel(r, sigma_s);
	Mat srcBorder;
	copyMakeBorder(src, srcBorder, r, r, r, r, BORDER_REPLICATE);
	Mat temp_32f(srcBorder);
	// ���ꂾ��3ch
	Mat dest_32f(src.rows, src.cols, CV_32FC3);
	const int step = srcBorder.cols;
	if (src.channels() == 3)
	{
		Mat splitImg[3];
		split(srcBorder, splitImg);
#pragma omp parallel for
		for (int y = 0; y < src.rows; y++)
		{
			// rptr������ptr�ɏ���������
			uchar* rptr = splitImg[0].ptr<uchar>(y + r);
			uchar* gptr = splitImg[1].ptr<uchar>(y + r);
			uchar* bptr = splitImg[2].ptr<uchar>(y + r);

			// ���ꂪSIMD�̌��ʊi�[�p�̃|�C���^
			float* dptr = dest_32f.ptr<float>(y, 0);

			for (int x = 0; x < src.cols; x += 8)
			{
				// tgt���Ƃ�(rrptr�Ƃ�����Ȃ��āArptr�Ƃ�)
				__m256 mtgt_r = _mm256_cvtepi32_ps(_mm256_cvtepu8_epi32(_mm_loadl_epi64((__m128i*)(rptr + x + r))));
				__m256 mtgt_g = _mm256_cvtepi32_ps(_mm256_cvtepu8_epi32(_mm_loadl_epi64((__m128i*)(gptr + x + r))));
				__m256 mtgt_b = _mm256_cvtepi32_ps(_mm256_cvtepu8_epi32(_mm_loadl_epi64((__m128i*)(bptr + x + r))));
				__m256 msum_r = _mm256_setzero_ps();
				__m256 msum_g = _mm256_setzero_ps();
				__m256 msum_b = _mm256_setzero_ps();
				__m256 mwsum = _mm256_setzero_ps();
				int kidx = 0;
				for (int i = -r; i <= r; i++)
				{
					unsigned char* rrptr = rptr + i * step;
					unsigned char* grptr = gptr + i * step;
					unsigned char* brptr = bptr + i * step;
					for (int j = -r; j <= r; j++)
					{
						// ��Ԃ̃K�E�V�A��
						__m256 mws = _mm256_set1_ps(kernel[kidx++]);

						__m256 mref_r = _mm256_cvtepi32_ps(_mm256_cvtepu8_epi32(_mm_loadl_epi64((__m128i*)(rrptr + x + j + r))));
						__m256 mref_g = _mm256_cvtepi32_ps(_mm256_cvtepu8_epi32(_mm_loadl_epi64((__m128i*)(grptr + x + j + r))));
						__m256 mref_b = _mm256_cvtepi32_ps(_mm256_cvtepu8_epi32(_mm_loadl_epi64((__m128i*)(brptr + x + j + r))));

						// tgt��ref�̍����Ƃ�
						__m256 sub_r = _mm256_sub_ps(mref_r, mtgt_r);
						__m256 sub_g = _mm256_sub_ps(mref_g, mtgt_g);
						__m256 sub_b = _mm256_sub_ps(mref_b, mtgt_b);
						// �����悷��
						sub_r = _mm256_mul_ps(sub_r, sub_r);
						sub_g = _mm256_mul_ps(sub_g, sub_g);
						sub_b = _mm256_mul_ps(sub_b, sub_b);
						// ���a���v�Z
						__m256 subsub_rgb = _mm256_add_ps(sub_r, _mm256_add_ps(sub_g, sub_b));

						// (-2.f * sigma_r * sigma_r)�̃x�N�g��������āA�����āAexp�ɓ���āAmwcolor�Ƃ��Ă݂�
						__m256 exp_div = _mm256_set1_ps(-2.f * sigma_r * sigma_r);
						__m256 mwcolor = _mm256_exp_ps(_mm256_div_ps(subsub_rgb, exp_div));

						// msum_r�Ƃ��̓{�b�N�X���̋P�x�̘a�ɁA�d��2��ނ����������̘̂a
						msum_r = _mm256_fmadd_ps(mref_r, _mm256_mul_ps(mws, mwcolor), msum_r);
						msum_g = _mm256_fmadd_ps(mref_g, _mm256_mul_ps(mws, mwcolor), msum_g);
						msum_b = _mm256_fmadd_ps(mref_b, _mm256_mul_ps(mws, mwcolor), msum_b);
						mwsum = _mm256_fmadd_ps(mws, mwcolor, mwsum);
					}
				}
				// �����ŏ����I�������f������A�o�C���e��������(2r+1)*(2r+1)�̋P�x�̘a��(2r+1)(2r+1)�̏d�݂̘a�Ŋ���������
				msum_r = _mm256_div_ps(msum_r, mwsum);
				msum_g = _mm256_div_ps(msum_g, mwsum);
				msum_b = _mm256_div_ps(msum_b, mwsum);
				_mm256_stream_ps_color(dptr + 3 * x, msum_r, msum_g, msum_b);
			}
		}
	}
	else if (src.channels() == 1)
	{
		//gray image
		std::cout << "not support gray image" << std::endl;
		return;
	}
	dest = dest_32f;
	//convertTo�͐؂�̂Ă���Ȃ��Ďl�̌ܓ�
	dest.convertTo(dest, CV_8UC3);
	_mm_free(kernel);
}

// SIMD����(�J�[�l�����[�v�W�J)
void BilateralFilter_SIMD_kernelloop(const Mat& src, Mat& dest, const int r, float sigma_r, float sigma_s)
{
	//generate Gaussian kernel
	float* kernel = createGaussianKernel(r, sigma_s);
	Mat srcBorder;
	copyMakeBorder(src, srcBorder, r, r, r, r, BORDER_REPLICATE);
	Mat temp_32f(srcBorder);
	// ���ꂾ��3ch
	Mat dest_32f(src.rows, src.cols, CV_32FC3);
	const int step = srcBorder.cols;

	// ���ʗp
	Mat splitdest[3];
	split(src, splitdest);

	if (src.channels() == 3)
	{
		Mat splitImg[3];
		split(srcBorder, splitImg);
		// �J�[�l�����[�v�W�J�ł̂��܂�
		const int kernel_amari = (2 * r + 1) % 8;
#pragma omp parallel for
		for (int y = 0; y < src.rows; y++)
		{
			// tgt�̏c�̍��W
			uchar* rptr = splitImg[0].ptr<uchar>(y + r);
			uchar* gptr = splitImg[1].ptr<uchar>(y + r);
			uchar* bptr = splitImg[2].ptr<uchar>(y + r);

			// �ŏI���ʊi�[�p
			uchar* kakunou_r = splitdest[0].ptr<uchar>(y);
			uchar* kakunou_g = splitdest[1].ptr<uchar>(y);
			uchar* kakunou_b = splitdest[2].ptr<uchar>(y);

			for (int x = 0; x < src.cols; x++)
			{
				// SIMD�p�̃x�N�g��
				__m256 msum_r_ref = _mm256_setzero_ps();
				__m256 msum_g_ref = _mm256_setzero_ps();
				__m256 msum_b_ref = _mm256_setzero_ps();

				__m256 msum_r = _mm256_setzero_ps();
				__m256 msum_g = _mm256_setzero_ps();
				__m256 msum_b = _mm256_setzero_ps();
				__m256 mwsum = _mm256_setzero_ps();

				// �X�J���p�̕ϐ�
				float sum[3] = { 0 };
				float wsum = 0;

				// ptr
				const float tgt_r = rptr[x + r];
				const float tgt_g = gptr[x + r];
				const float tgt_b = bptr[x + r];

				// ����͋�Ԃ̃K�E�V�A����SIMD�ƃX�J���ŃC���f�b�N�X���p
				int kidx = 0;
				for (int i = -r; i <= r; i++)
				{
					unsigned char* rrptr = rptr + i * step;
					unsigned char* grptr = gptr + i * step;
					unsigned char* brptr = bptr + i * step;
					// �J�[�l�����[�v�W�J����
					int j = -r;
					for (; j <= r - kernel_amari; j += 8)
					{
						// ws
						__m256 mws = _mm256_set_ps(kernel[kidx + 7], kernel[kidx + 6], kernel[kidx + 5], kernel[kidx + 4], kernel[kidx + 3], kernel[kidx + 2], kernel[kidx + 1], kernel[kidx]);
						kidx += 8;

						// �J�[�l���ŉE��8�Ƃ�(rrptr+x+j,rrptr+x+j+1,...,rrptr+x+j+7�ɂȂ肻��)
						__m256 mkernelref_r = _mm256_cvtepi32_ps(_mm256_cvtepu8_epi32(_mm_loadl_epi64((__m128i*)(rrptr + x + j + r))));
						__m256 mkernelref_g = _mm256_cvtepi32_ps(_mm256_cvtepu8_epi32(_mm_loadl_epi64((__m128i*)(grptr + x + j + r))));
						__m256 mkernelref_b = _mm256_cvtepi32_ps(_mm256_cvtepu8_epi32(_mm_loadl_epi64((__m128i*)(brptr + x + j + r))));

						// wcolor
						__m256 tgt_r_same = _mm256_set1_ps(tgt_r);
						__m256 tgt_g_same = _mm256_set1_ps(tgt_g);
						__m256 tgt_b_same = _mm256_set1_ps(tgt_b);

						// tgt��ref�̍����Ƃ�
						__m256 sub_r = _mm256_sub_ps(mkernelref_r, tgt_r_same);
						__m256 sub_g = _mm256_sub_ps(mkernelref_g, tgt_g_same);
						__m256 sub_b = _mm256_sub_ps(mkernelref_b, tgt_b_same);

						// �����悷��
						sub_r = _mm256_mul_ps(sub_r, sub_r);
						sub_g = _mm256_mul_ps(sub_g, sub_g);
						sub_b = _mm256_mul_ps(sub_b, sub_b);
						// ���a���v�Z
						__m256 subsub_rgb = _mm256_add_ps(sub_r, _mm256_add_ps(sub_g, sub_b));

						// (-2.f * sigma_r * sigma_r)�̃x�N�g��������āA�����āAexp�ɓ���āAmwcolor�Ƃ��Ă݂�
						__m256 exp_div = _mm256_set1_ps(-2.f * sigma_r * sigma_r);
						__m256 mwcolor = _mm256_exp_ps(_mm256_div_ps(subsub_rgb, exp_div));

						// mws,mwcolor���ł�������̂ŁA8����ref�ɁAws*wcolor�������đ���
						msum_r_ref = _mm256_fmadd_ps(mkernelref_r, _mm256_mul_ps(mws, mwcolor), msum_r_ref);
						msum_g_ref = _mm256_fmadd_ps(mkernelref_g, _mm256_mul_ps(mws, mwcolor), msum_g_ref);
						msum_b_ref = _mm256_fmadd_ps(mkernelref_b, _mm256_mul_ps(mws, mwcolor), msum_b_ref);

						// ��́Amws*mwcolor��mwsum�ɑ����Ă���
						mwsum = _mm256_fmadd_ps(mws, mwcolor, mwsum);
					}

					// �J�[�l�����[�v�̂��܂����E�[�̏���
					for (; j <= r; j++)
					{
						// ws
						float ws = kernel[kidx++];
						// wcolor
						const float ref_r = rrptr[x + r + j];
						const float ref_g = grptr[x + r + j];
						const float ref_b = brptr[x + r + j];

						const float range_distance = (ref_r - tgt_r) * (ref_r - tgt_r) + (ref_g - tgt_g) * (ref_g - tgt_g) + (ref_b - tgt_b) * (ref_b - tgt_b);
						const float wcolor = exp(-range_distance / (2.f * sigma_r * sigma_r));

						// ws*wcolor*�Q�Ƃ�r,g,b�𑫂�(sum[3]�ɍ���������)
						sum[0] += ws * wcolor * ref_r;
						sum[1] += ws * wcolor * ref_g;
						sum[2] += ws * wcolor * ref_b;
						// wsum�ɁAws*wcolor�𑫂�
						wsum += ws * wcolor;
					}
				}
				// �����ŏ����I�������f������A�o�C���e��������(2r+1)*(2r+1)�̋P�x�̘a��(2r+1)(2r+1)�̏d�݂̘a�Ŋ��������̂���ꂽ��
				// ��f���[�v�W�J���ƁASIMD�����ŏ������Ċ��ꂽ���ǁA�J�[�l�����Ƃ��܂肪�ł�̂�
				// msum_r,g,b��mwsum�ɁA�X�J���̕��̒l�����Ƃ����đ����Ȃ��ƃ_��

				// hadd�ŁAmsum_r_ref,msum_g_ref,msum_b_ref,mwsum�𑫂��āA�X�J���̕��Ɠ���
				msum_r_ref = _mm256_hadd_ps(msum_r_ref, msum_r_ref);
				msum_r_ref = _mm256_hadd_ps(msum_r_ref, msum_r_ref);
				sum[0] += ((float*)&msum_r_ref)[0] + ((float*)&msum_r_ref)[4];
				msum_g_ref = _mm256_hadd_ps(msum_g_ref, msum_g_ref);
				msum_g_ref = _mm256_hadd_ps(msum_g_ref, msum_g_ref);
				sum[1] += ((float*)&msum_g_ref)[0] + ((float*)&msum_g_ref)[4];
				msum_b_ref = _mm256_hadd_ps(msum_b_ref, msum_b_ref);
				msum_b_ref = _mm256_hadd_ps(msum_b_ref, msum_b_ref);
				sum[2] += ((float*)&msum_b_ref)[0] + ((float*)&msum_b_ref)[4];
				mwsum = _mm256_hadd_ps(mwsum, mwsum);
				mwsum = _mm256_hadd_ps(mwsum, mwsum);
				wsum += ((float*)&mwsum)[0] + ((float*)&mwsum)[4];


				// �ꉞat��ptr�ɏ��������Ċi�[
				kakunou_r[x] = sum[0] / wsum + 0.5f;
				kakunou_g[x] = sum[1] / wsum + 0.5f;
				kakunou_b[x] = sum[2] / wsum + 0.5f;

			}
		}
	}
	else if (src.channels() == 1)
	{
		//gray image
		std::cout << "not support gray image" << std::endl;
		return;
	}

	// �������ċ��߂�splitdest��merge����
	merge(splitdest, 3, dest_32f);
	dest = Mat(dest_32f);

	//convertTo�͐؂�̂Ă���Ȃ��Ďl�̌ܓ�
	dest.convertTo(dest, CV_8UC3);
	_mm_free(kernel);
}
