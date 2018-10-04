


// 获取指定像素点放射变换后的新的坐标位置
// https://blog.csdn.net/watkinsong/article/details/10212715#
CvPoint getPointAffinedPos(const CvPoint &src, const CvPoint ¢er, double angle)
{
	CvPoint dst;
	int x = src.x - center.x;
	int y = src.y - center.y;

	dst.x = cvRound(x * cos(angle) + y * sin(angle) + center.x);
	dst.y = cvRound(-x * sin(angle) + y * cos(angle) + center.y);
	return dst;
}
