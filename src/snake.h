#ifndef SNAKE_H
#define SNAKE_H
#include <QObject>
#include <list>
#include <QImage>
#include "my_user_types.h"

struct snakePoint{
    float x;
    float y;
};

class snake : public QObject
{
    Q_OBJECT
public:
    snake(QImage *const src, QImage *const dst, QObject *parent = 0);

    std::list<snakePoint> points;

    int moveAllPointsOnes();
    void addPoint(const snakePoint p);
    void addInterest(const rawArray2D<float> & src);
    void addInterest(rawArray2D<float> && src);
    void addInterestCoef(const rawArray2D<p2i> & src);
    void addInterestCoef(rawArray2D<p2i> && src);
    void addForce(const rawArray2D<float> & src);
    void addForce(rawArray2D<float> && src);
    void eliminateLoops(int n);
    void resizeSnake();
    void setMinimalDistance(const float dist);
    void setSearchDistance(const int dist);
    void projectToImage(rawArray2D<unsigned char> *const dst, unsigned char imprint);
    void signleSnakeImage(const rawArray2D<unsigned char> & src, rawArray2D<unsigned char> *const dst);

    snake() : points(), interest(), interestCoef(), force(),
        localInterest(), localInterestCoef(), localElasticity(), localCurv(), localForce(),
        radius(0), p2pMinDist(2.0f) { }

private:
    int movePointOnes(const std::list<snakePoint>::iterator it, snakePoint *const newpoint);
    void normalize(rawArray2D<float> *const area, const float norm, const int h, const int w, const float min_diap = 9999.99f);
    float elasticity_term(const std::list<snakePoint>::iterator it, const snakePoint p);
    float curvature_term(const std::list<snakePoint>::iterator it, const snakePoint p);
    float interest_term(const p2i p);
    float force_term(const p2i p);
    float interest_coef(const std::list<snakePoint>::iterator it, const snakePoint p);

public:
    rawArray2D<float> interest;
    int iter_lim;
private:
    rawArray2D<p2i> interestCoef;
    rawArray2D<float> force;
    float avgDist;
    rawArray2D<float> localInterest;
    rawArray2D<float> localInterestCoef;
    rawArray2D<float> localElasticity;
    rawArray2D<float> localCurv;
    rawArray2D<float> localForce;
    int radius;
    float p2pMinDist;
    QImage * srcImage;
    QImage * dstImage;
    float interest_strength;
    float interestCoef_strength;
    float elasticity_strength;
    float curvature_strength;
    float force_strength;
    float min_loop_distance;

public slots:
    void apply_snake();
signals:
    void image_ready();
    void print_message(const QString&);
    void print_progress(const int);

};

void signleSnakeImage(const rawArray2D<unsigned char> & src, rawArray2D<unsigned char> *const dst);

#endif // SNAKE_H
