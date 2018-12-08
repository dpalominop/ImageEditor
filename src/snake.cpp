#include "snake.h"
#include <cmath>

static const int last_to_print = 23000;

static const unsigned int Gaublu[5][5] = {
    {2,  4,  5,  4,  2},
    {4,  9, 12,  9,  4},
    {5, 12, 15, 12,  5},
    {4,  9, 12,  9,  4},
    {2,  4,  5,  4,  2},
};

static const signed int Gxm[3][3] = {
    {-1, 0, 1},
    {-2, 0, 2},
    {-1, 0, 1},
};

static const signed int Gym[3][3] = {
    {-1,-2,-1},
    { 0, 0, 0},
    { 1, 2, 1},
};

static float vector_length(const snakePoint vec)
{
    float buf = vec.x*vec.x + vec.y*vec.y;
    buf = sqrt(buf);
    return buf;
}

static float dotProjectToCos(const snakePoint vec1, const snakePoint vec2, const float vectorThresh)
{
    const float scalar = (float)(vec1.x*vec2.x + vec1.y*vec2.y);
    const float l1 = vector_length(vec1);
    const float l2 = vector_length(vec2);
    if ((l1 < vectorThresh) || (l2 < vectorThresh))
        return 0.99f;
    float cosval = (scalar/(vector_length(vec1)*vector_length(vec2)));
    cosval = cosval >  0.999f ?  0.999f : cosval;
    cosval = cosval < -0.999f ? -0.999f : cosval;
    return cosval;
}

static float angle(const snakePoint vec1, const snakePoint vec2, const float vectorThresh)
{
    const float ang = acos(dotProjectToCos(vec1, vec2, vectorThresh));
    return ang;
}//*/

snake::snake(QImage *const src, QImage *const dst, QObject *parent) : QObject(parent)
{
    srcImage = src;
    dstImage = dst;
    interest_strength =  3.0f;
    interestCoef_strength =  2.0f;
    elasticity_strength =  1.0f;
    curvature_strength =  2.0f;
    force_strength =  1.0f;
    iter_lim = 100;

    p2pMinDist = 3.0f;
    min_loop_distance = p2pMinDist;
    radius = 9;
    const int d = radius*2 + 1;
    localInterest.resize_without_copy(d, d);
    localInterestCoef.resize_without_copy(d, d);
    localElasticity.resize_without_copy(d, d);
    localCurv.resize_without_copy(d, d);
    localForce.resize_without_copy(d, d);

    return;
}

float snake::elasticity_term(const std::list<snakePoint>::iterator it, const snakePoint p)
{
    std::list<snakePoint>::const_iterator itp;
    if (it == points.cbegin())
        itp = points.cend();
    else
        itp = it;
    --itp;

    const float dy = p.y - itp->y;
    const float dx = p.x - itp->x;
    float elasticity = avgDist - sqrt(dy*dy+dx*dx);
    elasticity = elasticity*elasticity;

    return elasticity;
}

float snake::curvature_term(const std::list<snakePoint>::iterator it, const snakePoint p)
{
    std::list<snakePoint>::const_iterator itp;
    if (it == points.cbegin())
        itp = points.cend();
    else
        itp = it;
    --itp;

    std::list<snakePoint>::const_iterator itpp;
    if (itp == points.cbegin())
        itpp = points.cend();
    else
        itpp = itp;
    --itpp;

    snakePoint pp, ppp;
    pp.x = itp->x; pp.y = itp->y;
    ppp.x = itpp->x; ppp.y = itpp->y;

    //const float d2x = ppp.x - 2*pp.x + p.x;
    //const float d2y = ppp.y - 2*pp.y + p.y;

    snakePoint vec1, vec2;
    vec1.x = pp.x - ppp.x;
    vec1.y = pp.y - ppp.y;
    vec2.x = p.x - pp.x;
    vec2.y = p.y - pp.y;
    float ret = abs1f(angle(vec1, vec2, p2pMinDist/16));
    //float ret = d2x*d2x + d2y*d2y;

    return ret;
}

float snake::interest_term(const p2i p)
{
    const float buf = -interest(p.y, p.x);
    return buf;
}

float snake::force_term(const p2i p)
{
    const float buf = -force(p.y, p.x);
    return buf;
}

float snake::interest_coef(const std::list<snakePoint>::iterator it, const snakePoint p)
{
    std::list<snakePoint>::const_iterator itp;
    if (it == points.cbegin())
        itp = points.cend();
    else
        itp = it;
    --itp;

    snakePoint vec1, vec2;
    vec1.x = p.x - itp->x;
    vec1.y = p.y - itp->y;
    p2i vec2i = interestCoef((int)p.y, (int)p.x);

    if ((vec2i.x == 0)&(vec2i.y == 0))
        return 5.0f; //large value
    // else normalize this vector
    vec2.x = (float)vec2i.x;
    vec2.y = (float)vec2i.y;
    const float vec2L = sqrt(vec2.x*vec2.x + vec2.y*vec2.y);
    vec2.x *= p2pMinDist/vec2L;
    vec2.y *= p2pMinDist/vec2L;

    //return 1;
    const float res = abs1f(dotProjectToCos(vec1, vec2, p2pMinDist/4));
    return res;
    // maximum posssible value 1.0
}

int snake::movePointOnes(std::list<snakePoint>::iterator it, snakePoint *const newpoint)
{
    const int w = interest.width;
    const int h = interest.height;

    //float C = 1.0f;
    std::list<snakePoint>::iterator itp;
    if (it != points.begin())
    {
        itp = it;
        //C = 1.0f;
    }
    else
    {
        itp = points.end();
        //C = 0.0f;//0
    }
    --itp;


    p2f p0 = {it->x, it->y};
    //p2f pm1 = {itp->x, itp->y};
    p2i p = {(int)(p0.x + 0.5f), (int)(p0.y + 0.5f)};

    const int x1 = (p0.x > (radius - 1)) ? (int)(p0.x - radius + 0.5f) : 0;
    const int x2 = (p0.x < (w - 1 - radius)) ? (int)(p0.x + radius + 0.5f) : w - 1;
    const int y1 = (p0.y > (radius - 1)) ? (int)(p0.y - radius + 0.5f) : 0;
    const int y2 = (p0.y < (h - 1 - radius)) ? (int)(p0.y + radius + 0.5f) : h - 1;

    const int lw = x2 - x1 + 1;
    const int lh = y2 - y1 + 1;

    for (p.y = y1; p.y <= y2; ++p.y)
        for (p.x = x1; p.x <= x2; ++p.x)
        {
            snakePoint pf = {(float)p.x, (float)p.y};
            localInterest(p.y - y1, p.x - x1) = interest_term(p);
            localElasticity(p.y - y1, p.x - x1) = elasticity_term(it, pf);
            localCurv(p.y - y1, p.x - x1) = curvature_term(it, pf);
            localForce(p.y - y1, p.x - x1) = force_term(p);
            localInterestCoef(p.y - y1, p.x - x1) = interest_coef(it, pf);
        }
    normalize(&localInterest, interest_strength, lh, lw, 60.0f);
    normalize(&localElasticity, elasticity_strength, lh, lw, 0.1f);
    normalize(&localCurv, curvature_strength, lh, lw, 0.1f);
    normalize(&localForce, force_strength, lh, lw, 0.1f);
    normalize(&localInterestCoef, interestCoef_strength, lh, lw, 0.1f);


    newpoint->x = p0.x;
    newpoint->y = p0.y;
    p.x = (int)(p0.x + 0.5f);
    p.y = (int)(p0.y + 0.5f);
    const int lowInitialExtEnergy = (localInterest(p.y - y1, p.x - x1) < (0.2f*interest_strength));
    float bufIntE = localInterest(p.y - y1, p.x - x1);
    float bufIntCoefE = localInterestCoef(p.y - y1, p.x - x1);
    float bufElasE = localElasticity(p.y - y1, p.x - x1);
    float bufCurvE = localCurv(p.y - y1, p.x - x1);
    float bufForceE = localForce(p.y - y1, p.x - x1);
    //minIntE = bufIntE;
    //minIntCoefE = bufIntCoefE;
    //minElasE = bufElasE;
    //minCurvE = bufCurvE;
    //minForceE = bufForceE;
    float minE = 100000+bufIntE*bufIntE + bufIntCoefE*bufIntCoefE + bufElasE*bufElasE + bufCurvE*bufCurvE + bufForceE*bufForceE;
    for (p.y = y1; p.y <= y2; ++p.y)
        for (p.x = x1; p.x <= x2; ++p.x)
        {
            //if ((abs(itp->x - p.x) < p2pMinDist)&(abs(itp->y - p.y) < p2pMinDist))
            //    continue;

            bufIntE = localInterest(p.y - y1, p.x - x1);
            bufIntCoefE = localInterestCoef(p.y - y1, p.x - x1)*(lowInitialExtEnergy);
            bufElasE = localElasticity(p.y - y1, p.x - x1);
            bufCurvE = localCurv(p.y - y1, p.x - x1)*(lowInitialExtEnergy);
            bufForceE = localForce(p.y - y1, p.x - x1);
            const float newE = bufIntE*bufIntE + bufIntCoefE*bufIntCoefE + bufElasE*bufElasE + bufCurvE*bufCurvE + bufForceE*bufForceE;
            if (newE < minE)
            {
                minE = newE;
                newpoint->y = p.y;
                newpoint->x = p.x;
                //minIntE = bufIntE;
                //minIntCoefE = bufIntCoefE;
                //minElasE = bufElasE;
                //minCurvE = bufCurvE;
                //minForceE = bufForceE;
            }

        }

    return (int)sqrt((newpoint->x - p0.x)*(newpoint->x - p0.x) + (newpoint->y - p0.y)*(newpoint->y - p0.y));


    //return ((newpoint->x == it->x)&(newpoint->y == it->y));
}

int snake::moveAllPointsOnes()
{
    int moves = 0;
    //std::list<snakePoint> newpoints;
    snakePoint newp;
    int pn = 0;
    for (std::list<snakePoint>::iterator it = ++points.begin(); it != points.end();/*increment iterator only if new point accepted*/)
    {
        try
        {
//          if (pn == 1050)
//              std::cout << "start" << std::endl;
            newp = *it;
            std::list<snakePoint>::iterator itp;
            if (it != points.begin()) itp = it;
            else itp = points.end();
            --itp;
            //snakePoint prevp = *itp;
            moves += movePointOnes(it, &newp);

            // if point's new location is too close to previos neighbour point, remove it
            if ((abs(itp->x - newp.x) < p2pMinDist)&(abs(itp->y - newp.y) < p2pMinDist))
            {
                //std::cout << "erase" << std::endl;
                it = points.erase(it);
            }
            // else move to new location
            else
            {
                //newpoints.push_back(newp);
                *it = newp;
                ++pn;
                ++it;
            }


        }
        catch (std::string err)
        {
            std::cout << err << "<<" << pn << ">>"<< std::endl;
            exit(0);
        }
        catch (...)
        {
            std::cout << "somerror" << "<<" << pn << ">>"<< std::endl;
            exit(0);
        }
    }
    std::list<snakePoint>::iterator itb = points.begin();
    std::list<snakePoint>::iterator ite = --points.end();
    //snakePoint vec {ite->x - itb->x, ite->y - itb->y};
    if ((abs(ite->x - itb->x) < p2pMinDist)&(abs(ite->y - itb->y) < p2pMinDist))
        points.pop_back();
    //points = std::move(newpoints);
    return moves;
}

void snake::addPoint(const snakePoint p)
{
    if ((p.x < 0.1f)|(p.x >= (interest.width - 0.1f))|(p.y < 0.1f)|(p.y >= (interest.height - 0.1f)))
        return; // silently no addition if no interest specified for the point

    points.push_back(p);

    return;
}

void snake::normalize(rawArray2D<float> *const area, const float norm, const int h, const int w, const float min_diap)
{
    float min = (*area)(0, 0);
    float max = min;
    for (int y = 0; y < h; ++y)
        for (int x = 0; x < w; ++x)
        {
            const float val = (*area)(y, x);
            min = (val < min) ? val : min;
            max = (val > max) ? val : max;
        }

    min = (max - min) < min_diap ? max - min_diap : min;
    const float range = max - min;
    for (int y = 0; y < h; ++y)
        for (int x = 0; x < w; ++x)
        {
            float buf = (*area)(y,x);
            buf = buf - min;
            buf = buf / range;
            buf = buf * norm;
            (*area)(y, x) = buf;
        }
    return;
}

void snake::addInterest(const rawArray2D<float> & src)
{
    interest = src;
    return;
}

void snake::addInterest(rawArray2D<float> && src)
{
    interest = std::move(src);
    return;
}

void snake::addInterestCoef(const rawArray2D<p2i> & src)
{
    interestCoef = src;
    return;
}

void snake::addInterestCoef(rawArray2D<p2i> && src)
{
    interestCoef = std::move(src);
    return;
}

void snake::addForce(const rawArray2D<float> & src)
{
    force = src;
    return;
}

void snake::addForce(rawArray2D<float> && src)
{
    force = std::move(src);
    return;
}

static std::list<snakePoint>::iterator move_iterator_by(std::list<snakePoint>::iterator it, int n, std::list<snakePoint>::iterator first, std::list<snakePoint>::iterator last)
{
    std::list<snakePoint>::iterator plast = last;
    --plast;
    while (n > 0)
    {
        if (it == plast)
            //return it;
            it = first;
        else
            ++it;
        --n;
    }
    while (n < 0)
    {
        if (it == first)
            //return it;
            it = plast;
        else
            --it;
        ++n;
    }
    return it;
}

void erase_loop(std::list<snakePoint> *const points, std::list<snakePoint>::iterator start, std::list<snakePoint>::iterator end)
{
    std::list<snakePoint>::iterator pend = end; --pend;
    std::list<snakePoint>::iterator i = start;
    while (i != end)
    {
        if (i == points->end())
            i = points->begin();
        snakePoint p{i->x, i->y}; ///?
        i = points->erase(i);
    }
    return;
}

void snake::eliminateLoops(int n)
{
    //return;
    int sz = points.size();

    //remove loops
//*
    auto it = points.begin();
    it = move_iterator_by(it, 3, points.begin(), points.end());
    int pn = 3;
    for (; it != --points.end(); ++it)
    {
        if (n < 1) break;
        auto itp = it;  --itp;
        int dt = -sz/2;
        for(auto i = move_iterator_by(it, -sz/2, points.begin(), points.end()); i != itp; ++i)
        {
            if ((abs(i->x - it->x) < 1.9f*min_loop_distance)&(abs(i->y - it->y) < 1.9f*min_loop_distance))
            {
                //points.erase(i, it);
                erase_loop(&points, i, it);
                sz = points.size();
                std::cout << "loop" << std::endl;
                pn += dt;
                --n;
                break;
            }
            ++dt;
        }
        ++pn;
    }
//*/
    return;
}

void snake::resizeSnake()
{
    const float dist = p2pMinDist*2;

    //remove loops
/*
    auto it = points.begin();
    it = move_iterator_by(it, 3, points.begin(), points.end());
    int pn = 3;
    for (; it != --points.end(); ++it)
    {
        auto itp = it;  --itp;
        int dt = -sz/2;
        for(auto i = move_iterator_by(it, -sz/2, points.begin(), points.end()); i != itp; ++i)
        {
            if ((abs(i->x - it->x) < 1.9f*p2pMinDist)&(abs(i->y - it->y) < 1.9f*p2pMinDist))
            {
                //points.erase(i, it);
                erase_loop(&points, i, it);
                sz = points.size();
                std::cout << "loop" << std::endl;
                pn += dt;
                break;
            }
            ++dt;
        }
        ++pn;
    }
//*/

    std::list<snakePoint>::const_iterator p1 = points.cbegin();
    std::list<snakePoint>::iterator p2 = points.begin();; ++p2;
    while(p2 != points.end())
    {
        const float dx = p2->x - p1->x;
        const float dy = p2->y - p1->y;

        if ((abs1f(dx) > dist)||(abs1f(dy) > dist))
        {
            const snakePoint newPoint{p1->x + dx/2, p1->y + dy/2};
            p2 = points.insert(p2, newPoint);
        }
        else
        {
            ++p1;
            ++p2;
        }
    }

    // and for tails
    p1 = --points.end();
    p2 = points.begin();
    const float dx = p2->x - p1->x;
    const float dy = p2->y - p1->y;
    if ((abs(dx) > dist)||(abs(dy) > dist))
    {
        const int steps = (int)(vector_length({dx, dy}) / p2pMinDist + 0.5f);
        const float dxf = dx / steps;
        const float dyf = dy / steps;

        for (int i = 0; i < steps; ++i)
        {
            const float xf = p1->x + dxf*i;
            const float yf = p1->y + dyf*i;
            const snakePoint newPoint{xf, yf};
            points.push_back(newPoint);
        }
    }

    float sum = 0.0f;
    const std::list<snakePoint>::const_iterator cit_end = points.end();
    for (std::list<snakePoint>::const_iterator cit2 = ++points.cbegin(); cit2 != cit_end; ++cit2)
    {
        std::list<snakePoint>::const_iterator cit1 = cit2; --cit1;
        const float dy = cit2->y - cit1->y;
        const float dx = cit2->x - cit1->x;
        sum += sqrt(dy*dy+dx*dx);
    }
    avgDist = sum / points.size();

    return;
}

void snake::projectToImage(rawArray2D<unsigned char> *const dst, unsigned char imprint)
{
    //const int sz = interest.height*interest.width;
    const int h = interest.height;
    const int w = interest.width;

    int i = 0;
    for (std::list<snakePoint>::const_iterator p1 = points.begin(); p1 != --points.end(); ++p1)
    {
        (*dst)(p1->y, p1->x) = imprint;
        if ((p1->x > 2)&(p1->y > 2)&(p1->x < (w - 4))&(p1->y < (h - 4)))
        {
            (*dst)(p1->y + 3, p1->x) = imprint;
            (*dst)(p1->y + 2, p1->x) = imprint;
            (*dst)(p1->y + 1, p1->x) = imprint;
            (*dst)(p1->y, p1->x - 3) = imprint;
                (*dst)(p1->y, p1->x - 2) = imprint;
                    (*dst)(p1->y, p1->x - 1) = imprint;
                        (*dst)(p1->y, p1->x + 1) = imprint;
                            (*dst)(p1->y, p1->x + 2) = imprint;
                                (*dst)(p1->y, p1->x + 3) = imprint;
            (*dst)(p1->y - 1, p1->x) = imprint;
            (*dst)(p1->y - 2, p1->x) = imprint;
            (*dst)(p1->y - 3, p1->x) = imprint;
        }
        std::list<snakePoint>::const_iterator p2 = p1; ++p2;

        const float dx = p2->x - p1->x;
        const float dy = p2->y - p1->y;
        if ((abs1f(dx) > 1)||(abs1f(dy) > 1))
        {
            const int steps = (int)(vector_length({dx, dy}) + 0.5f);
            const float dxf = dx / steps;
            const float dyf = dy / steps;

            for (int i = 1; i < steps; ++i)
            {
                const float xf = p1->x + dxf*i;
                const float yf = p1->y + dyf*i;
                const p2i pn{(int)xf, (int)yf};
                (*dst)(pn.y, pn.x) = imprint;
            }
        }
        ++i;
        if (i >= last_to_print)
            break;
    }

    /*const std::list<snakePoint>::const_iterator p = --points.end();
    (*dst)(p->y, p->x) = imprint;
    if ((p->x > 2)&(p->y > 2)&(p->x < (w - 4))&(p->y < (h - 4)))
    {
          (*dst)(p->y + 3, p->x) = imprint;
          (*dst)(p->y + 2, p->x) = imprint;
          (*dst)(p->y + 1, p->x) = imprint;
(*dst)(p->y, p->x - 3) = imprint;
    (*dst)(p->y, p->x - 2) = imprint;
        (*dst)(p->y, p->x - 1) = imprint;
            (*dst)(p->y, p->x + 1) = imprint;
                (*dst)(p->y, p->x + 2) = imprint;
                    (*dst)(p->y, p->x + 3) = imprint;
          (*dst)(p->y - 1, p->x) = imprint;
          (*dst)(p->y - 2, p->x) = imprint;
          (*dst)(p->y - 3, p->x) = imprint;
    } */

    return;
}

void snake::signleSnakeImage(const rawArray2D<unsigned char> & src, rawArray2D<unsigned char> *const dst)
{
    // first gradient
    const int w = (int)src.width;
    const int h = (int)src.height;
    rawArray2D<float> gradient(h, w);
    rawArray2D<float> emptyarr(h, w);
    rawArray2D<p2i> gradient_dir(h, w);
    dst->resize_without_copy(h, w);
    //memset(dst->data, 0, w*h);
    *dst = src;
    rawArray2D<unsigned char> temp(h, w);

    // Gaussian blur
    for (int y = 0; y < h; ++y)
        for (int x = 0; x < w; ++x)
        {
            int dx[5], dy[5];
            dx[1] = -(x>0);
            dx[0] = (x > 1) ? -2 : dx[1];
            dx[2] = 0;
            dx[3] = (x < (w - 1));
            dx[4] = (x < (w - 2)) ? +2 : dx[3];
            dy[1] = -(y > 0);
            dy[0] = (y > 1) ? -2 : dy[1];
            dy[2] = 0;
            dy[3] = (y < (h - 1));
            dy[4] = (y < (h - 2)) ? +2 : dy[3];
            unsigned int pix = 0;
            for (int i = 0; i < 5; ++i)
                for (int j = 0; j < 5; ++j)
                    pix += src(y+dy[i], x+dx[i])*Gaublu[i][j];
            pix /= 159;
            temp(y, x) = (unsigned char)(pix);
        }

    // applying sobel mask
    for (int y = 0; y < h; ++y)
        for (int x = 0; x < w; ++x)
        {
            int dx[3], dy[3];
            dx[0] = -(x>0);
            dx[1] = 0;
            dx[2] = (x<(w-1));
            dy[0] = -(y>0);
            dy[1] = 0;
            dy[2] = (y<(h-1));
            int Gx = 0, Gy = 0;
            for (int i = 0; i < 3; ++i)
                for (int j = 0; j < 3; ++j)
                {
                    Gx += temp(y+dy[i], x+dx[j]) * Gxm[i][j];
                    Gy += temp(y+dy[i], x+dx[j]) * Gym[i][j];
                }
            gradient(y, x) = sqrt(Gx*Gx + Gy*Gy);
            gradient_dir(y, x) = {Gx, Gy};
        }

    addInterest(std::move(gradient));
    addInterestCoef(std::move(gradient_dir));

    memset(emptyarr.data, 0, h*w*sizeof(float));
    addForce(std::move(emptyarr));

    //*
    const float de = 110.0f;
    const float wf = w;
    const float hf = h;
    points.clear();
    addPoint({0 + de, 0 + de});
    addPoint({wf/4, 0 + de});
    addPoint({wf/2, 0 + de});
    addPoint({3*wf/4, 0 + de});
    addPoint({wf - 1 - de, 0 + de});
    addPoint({wf - 1 - de, hf/4});
    addPoint({wf - 1 - de, hf/2});
    addPoint({wf - 1 - de, 3*hf/4});
    addPoint({wf - 1 - de, hf - 1 - de});
    addPoint({3*wf/4, hf - 1 - de});
    addPoint({wf/2, hf - 1 - de});
    addPoint({wf/4, hf - 1 - de});
    addPoint({0 + de, hf - 1 - de});
    addPoint({0 + de, 3*hf/4});
    addPoint({0 + de, hf/2});
    addPoint({0 + de, hf/4});
    addPoint({0 + de/2, 0 + de});
    addPoint({0 + 3*de/4, 0 + de});//*/

    resizeSnake();
    projectToImage(dst, 30);

    int prev_moves, moves = 1000, i = 0;
    const float imprint = (225.0f) / iter_lim;

    int percent = 0;
    int prev_percent = percent;

    while (1)
    {
        ++i;
        prev_moves = moves;
        if (moves == 0)
            std::cout << "done" << std::endl;
        moves = moveAllPointsOnes();
        const int sz = points.size();
        eliminateLoops(100);
        resizeSnake();
        const int sz2 = points.size();
        std::cout << "iter=" << i << "   moves=" << moves << "   size=" << sz << "   resized=" << sz2 << std::endl;
        projectToImage(dst, (unsigned char)(i * imprint));
        if (sz2 < 7)
            break;
        if (prev_moves == 0)
            break;
        if (i >= iter_lim)
            break;

        percent = i*20/iter_lim;
        if (percent != prev_percent)
        {
            emit print_progress(percent * 5);
            prev_percent = percent;
        }

    }

    projectToImage(dst, 255);
    return;
}

void snake::apply_snake()
{
    emit print_progress(0);
    emit print_message(QString("snake creeping..."));
    const int w = srcImage->width();
    const int h = srcImage->height();
    rawArray2D<unsigned char> srcFrame(h, w);
    rawArray2D<unsigned char> dstFrame(h, w);
    *dstImage = QImage(w, h, QImage::Format_RGB32);

    for(int y = 0; y < h; ++y)
        for(int x = 0; x < w; ++x)
        {
            QRgb pix = srcImage->pixel(x, y);
            const int luma32 = 0.299 * qRed(pix) + 0.587 * qGreen(pix) + 0.114 * qBlue(pix);
            const unsigned char luma8 = luma32 > 255 ? 255 : luma32 < 0 ? 0 : luma32;
            srcFrame(y, x) = luma8;
            dstImage->setPixel(x, y, pix);
        }
    *dstImage = QImage(w, h, QImage::Format_RGB32);

    signleSnakeImage(srcFrame, &dstFrame);

    for(int y = 0; y < h; ++y)
        for(int x = 0; x < w; ++x)
        {
            const unsigned char level  = dstFrame(y, x);
            QRgb pix = qRgb(level, level, level);
            dstImage->setPixel(x, y, pix);
        }

    emit image_ready();
    emit print_progress(100);
    emit print_message(QString("snake creeping...finished"));

    return;
}



















