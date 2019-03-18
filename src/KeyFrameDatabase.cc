/**
* This file is part of ORB-SLAM2.
*
* Copyright (C) 2014-2016 Raúl Mur-Artal <raulmur at unizar dot es> (University of Zaragoza)
* For more information see <https://github.com/raulmur/ORB_SLAM2>
*
* ORB-SLAM2 is free software: you can redistribute it and/or modify
* it under the terms of the GNU General Public License as published by
* the Free Software Foundation, either version 3 of the License, or
* (at your option) any later version.
*
* ORB-SLAM2 is distributed in the hope that it will be useful,
* but WITHOUT ANY WARRANTY; without even the implied warranty of
* MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
* GNU General Public License for more details.
*
* You should have received a copy of the GNU General Public License
* along with ORB-SLAM2. If not, see <http://www.gnu.org/licenses/>.
*/

#include "KeyFrameDatabase.h"

#include "KeyFrame.h"
#include "Thirdparty/DBoW2/DBoW2/BowVector.h"

#include <mutex>

using namespace std;

namespace ORB_SLAM2
{

KeyFrameDatabase::KeyFrameDatabase (const ORBVocabulary &voc):
    mpVoc(&voc)
{
    mvInvertedFile.resize(voc.size()); // voc.size() 字典中有多少个单词！
}

//! \brief 在关键帧数据库{用来重定位/闭环检测}中插入新的关键帧。(按照 Bow 向量对应的单词 id 分类添加)
//! \note  这里的关键帧都是在局部建图线程中计算了其 Bow 有关的变量。
//!        然后在闭环线程中将关键帧添加到数据库，为了后面的闭环检测
//!        计算关键帧和数据库中历史关键帧的得分，进而决定备选的闭环帧）
void KeyFrameDatabase::add(KeyFrame *pKF)
{
    unique_lock<mutex> lock(mMutex);

    for(DBoW2::BowVector::const_iterator vit= pKF->mBowVec.begin(), vend=pKF->mBowVec.end(); vit!=vend; vit++)
        mvInvertedFile[vit->first].push_back(pKF); // 在对应的 id 项中添加该关键帧
}
// 在关键帧数据库中擦除关键帧 pKF。按照单词索引进行擦除
void KeyFrameDatabase::erase(KeyFrame* pKF)
{
    unique_lock<mutex> lock(mMutex);

    // Erase elements in the Inverse File for the entry // 对关键帧 pKF 包含的单词进行遍历
    for(DBoW2::BowVector::const_iterator vit=pKF->mBowVec.begin(), vend=pKF->mBowVec.end(); vit!=vend; vit++)
    {
        // List of keyframes that share the word
        list<KeyFrame*> &lKFs =   mvInvertedFile[vit->first];

        for(list<KeyFrame*>::iterator lit=lKFs.begin(), lend= lKFs.end(); lit!=lend; lit++) // 当前单词 id 对应的所有关键帧
        {
            if(pKF==*lit)
            {
                lKFs.erase(lit); // 将关键帧 pKF 擦除
                break;
            }
        }
    }
}

void KeyFrameDatabase::clear()
{
    mvInvertedFile.clear();
    mvInvertedFile.resize(mpVoc->size());
}

//! \see VII LOOP CLOSING A.Loop Candidates Detection
//! \brief 在关键帧数据库中找到与 pKF 关键帧可能构成闭环的关键帧集。(共同存在单词个数作为两个关键帧相似性依据)
//! \param pKF 当前闭环线程正在处理的关键帧
//! \param minScore pKF 关键帧与其所有临近关键帧得到的最低相似性得分作为筛选阈值
//! \return 检测到的潜在闭环帧集(能够与 pKF 关键帧构成一个闭环) vpLoopCandidates

//! \details
//!   1）给定一个闭环线程正在处理的关键帧 pKF。找到不属于其临近关键帧且与其有共同单词个数的所有关键帧(利用关键帧数据库) 加入到这个变量 list<KeyFrame*> lKFSharingWords
//!   2) 在 lKFSharingWords 中，遍历所有关键帧。找到与当前关键帧 pKF 最多共同单词个数。利用最多共同单词个数的 80% 作为最低共同单词个数阈值 minCommonWords。
//!   3）选择 lKFSharingWords 中关键帧与当前处理的关键帧 pKF 共同单词个数 > 最低阈值(minCommonWords)的关键帧。并计算与 pKF 的相似得分。
//!      对于得分 > minScore 的关键帧加入 list<pair<float,KeyFrame*> > lScoreAndMatch 里
//!   4）在 lScoreAndMatch 中，遍历选择一个关键帧。从其 10 个临近关键帧集中选择 满足条件 {共同单词个数 > 最低阈值(minCommonWords)，得分 > minScore 的临近关键帧}时，
//!      该 10 个临近关键帧与当前关键帧 pKF 的累计相似性得分。找到其中的最多得分及其对应关键帧加入到 list<pair<float,KeyFrame*> > lAccScoreAndMatch;
//!      遍历完 lScoreAndMatch 包含的所有的关键帧后。选择最高的累计得分 bestAccScore;
//!   5）通过 4）计算得到的最高累计得分(bestAccScore)，计算最低累计得分阈值 ：minScoreToRetain = 0.75f*bestAccScore;
//!      然后利用这个阈值在 4) 中得到的 lAccScoreAndMatch 中选择满足阈值的关键帧。加入到 vpLoopCandidates 变量中。作为潜在的闭环帧返回
//! 这里的操作实际上对于小闭环的查找成功率较小。可能为了避免频繁的闭环消耗计算资源。
vector<KeyFrame*> KeyFrameDatabase::DetectLoopCandidates(KeyFrame* pKF, float minScore)
{
    set<KeyFrame*> spConnectedKeyFrames = pKF->GetConnectedKeyFrames(); // 返回与当前关键帧链接的所有关键帧（未排序的）
    list<KeyFrame*> lKFsSharingWords; // 记录与当前关键帧 pKF 有着共同单词的所有关键帧。这些关键帧不包含 pKF 的临近关键帧。

    // Search all keyframes that share a word with current keyframes
    // Discard keyframes connected to the query keyframe
    // 计算关键帧数据库中与当前关键帧 pKF 有共同单词的所有关键帧，并记录共同单词的个数。但是这个关键帧不包括当前关键帧 pKF 的所有临近关键帧
    {
        unique_lock<mutex> lock(mMutex);

        for(DBoW2::BowVector::const_iterator vit=pKF->mBowVec.begin(), vend=pKF->mBowVec.end(); vit != vend; vit++)
        {
            list<KeyFrame*> &lKFs =   mvInvertedFile[vit->first]; // vit->first 是单词 id，这里其实是获取单词 id 中包含的所有关键帧

            for(list<KeyFrame*>::iterator lit=lKFs.begin(), lend= lKFs.end(); lit!=lend; lit++)
            {
                KeyFrame* pKFi=*lit;
                if(pKFi->mnLoopQuery!=pKF->mnId) // 没有记录过
                {
                    pKFi->mnLoopWords=0;
                    if(!spConnectedKeyFrames.count(pKFi)) // 不能是当前关键帧 pKF 的临近关键帧。因为这里是为了找到潜在的备选关键帧
                    {
                        pKFi->mnLoopQuery=pKF->mnId;
                        lKFsSharingWords.push_back(pKFi); // 记录与当前关键帧 pKF 有着共同单词的所有关键帧。这些关键帧不包含 pKF 的临近关键帧。
                    }
                }
                pKFi->mnLoopWords++; // 说明 pKFi 关键帧与当前关键帧找到共同的单词个数 ++。
            }
        }
    }

    if(lKFsSharingWords.empty())
        return vector<KeyFrame*>();

    list<pair<float,KeyFrame*> > lScoreAndMatch; // >= 最低得分(minScore) 的关键帧及其对应的得分  <得分，对应的关键帧>

    // Only compare against those keyframes that share enough words
    int maxCommonWords=0;
    // 在与当前关键帧有共同单词的所有关键帧中(不包括临近关键帧)。计算与当前关键帧 pKF 最多的共同单词个数，
    for(list<KeyFrame*>::iterator lit=lKFsSharingWords.begin(), lend= lKFsSharingWords.end(); lit!=lend; lit++)
    {
        if((*lit)->mnLoopWords>maxCommonWords)
            maxCommonWords=(*lit)->mnLoopWords;
    }

    int minCommonWords = maxCommonWords*0.8f; // 80% 阈值

    int nscores=0; // 与当前正在处理的关键帧 pKF 有着共同单词个数的所有关键帧中(不属于 pKF 的临近关键帧)。满足共同单词个数 > 最低阈值条件的关键帧的个数

    // Compute similarity score. Retain the matches whose score is higher than minScore
    for(list<KeyFrame*>::iterator lit=lKFsSharingWords.begin(), lend= lKFsSharingWords.end(); lit!=lend; lit++)
    {
        KeyFrame* pKFi = *lit;

        if(pKFi->mnLoopWords>minCommonWords)
        {
            nscores++; // 记录满足该条件的关键帧个数

            float si = mpVoc->score(pKF->mBowVec,pKFi->mBowVec);

            pKFi->mLoopScore = si;
            if(si>=minScore) // 记录满足最低分的关键帧及其相似得分
                lScoreAndMatch.push_back(make_pair(si,pKFi));
        }
    }

    if(lScoreAndMatch.empty())
        return vector<KeyFrame*>();

    list<pair<float,KeyFrame*> > lAccScoreAndMatch; // <累计得分，对应的最好的关键帧> (accScore,pBestKF)，这里可能会有重复的元素！能否用 set??
    float bestAccScore = minScore;

    // Lets now accumulate score by covisibility。
    // 在上面符合最低得分的关键帧集中。遍历选取一个关键帧。计算该关键帧的临近关键帧集（找到满足最低得分条件的关键帧）与当前正在处理的关键帧 pKF 的累计得分
    for(list<pair<float,KeyFrame*> >::iterator it=lScoreAndMatch.begin(), itend=lScoreAndMatch.end(); it!=itend; it++)
    {
        KeyFrame* pKFi = it->second; // 计算该关键帧的临近关键帧集（找到满足最低得分条件的关键帧）与当前正在处理的关键帧 pKF 的累计得分
        vector<KeyFrame*> vpNeighs = pKFi->GetBestCovisibilityKeyFrames(10);

        float bestScore = it->first; // 计算 pKFi 临近关键帧集中，与当前正在处理的关键帧 pKF 最高得分
        float accScore = it->first; // 最高得分对应的关键帧
        KeyFrame* pBestKF = pKFi;
        for(vector<KeyFrame*>::iterator vit=vpNeighs.begin(), vend=vpNeighs.end(); vit!=vend; vit++)
        {
            KeyFrame* pKF2 = *vit;
            if(pKF2->mnLoopQuery==pKF->mnId && pKF2->mnLoopWords>minCommonWords) // 保证与当前正在处理的关键帧 pKF 有共同的单词。以及满足最低单词个数。
            {                                                                    // 这个条件实际上就是在 lScoreAndMatch 里面选择关键帧。
                accScore+=pKF2->mLoopScore; // 累计得分
                if(pKF2->mLoopScore>bestScore)
                {
                    pBestKF=pKF2;
                    bestScore = pKF2->mLoopScore;
                }
            }
        }

        lAccScoreAndMatch.push_back(make_pair(accScore,pBestKF));
        if(accScore>bestAccScore)
            bestAccScore=accScore;
    }

    // Return all those keyframes with a score higher than 0.75*bestScore
    float minScoreToRetain = 0.75f*bestAccScore;

    set<KeyFrame*> spAlreadyAddedKF;
    vector<KeyFrame*> vpLoopCandidates; // 返回值，将要返回的潜在的闭环关键帧集
    vpLoopCandidates.reserve(lAccScoreAndMatch.size());

    for(list<pair<float,KeyFrame*> >::iterator it=lAccScoreAndMatch.begin(), itend=lAccScoreAndMatch.end(); it!=itend; it++)
    {
        if(it->first>minScoreToRetain)
        {
            KeyFrame* pKFi = it->second;
            if(!spAlreadyAddedKF.count(pKFi))
            {
                vpLoopCandidates.push_back(pKFi);
                spAlreadyAddedKF.insert(pKFi);
            }
        }
    }


    return vpLoopCandidates;
}

// 重定位检测。给定一个普通帧，此时处于追踪失败。该函数首先找到与当前帧有着共同单词的关键帧及其共同单词个数。
// 按照共同单词最大个数的 80% 作为一个阈值。利用这个阈值。选择与当前帧有着较多共同单词的关键帧并计算记录关键帧和当前帧的相似性评分（词袋向量）。
// 即这个集合为 list<pair<float,KeyFrame*> > lScoreAndMatch;
// 根据上面变量的记录。遍历内部每个元素(以关键帧为主导)。对于每个关键帧都找到与其在共视图中的 10 个好的临近关键帧组。
// 计算 10 个临近关键帧中与当前 Frame F 的累计评分。找到与当前帧相似评分（最大的）、累计评分及其对应的关键帧。
// 将累计得分和对应的最好的关键帧配对记录到下面变量中 {list<pair<float,KeyFrame*> > lAccScoreAndMatch; // 累计得分及其对应最高分的关键帧 }
// 最后根据累计评分阈值，在该变量中选择累计评分较高的--->关键帧。构成关键帧集返回来。
vector<KeyFrame*> KeyFrameDatabase::DetectRelocalizationCandidates(Frame *F)
{
    list<KeyFrame*> lKFsSharingWords; // 记录与当前 F 帧有共同单词的关键帧。没有按照共有单词数量排序

    // Search all keyframes that share a word with current frame
    {
        unique_lock<mutex> lock(mMutex);

        for(DBoW2::BowVector::const_iterator vit=F->mBowVec.begin(), vend=F->mBowVec.end(); vit != vend; vit++)
        {
            list<KeyFrame*> &lKFs =   mvInvertedFile[vit->first]; // 取出对应同一个单词 id 的所有关键帧

            for(list<KeyFrame*>::iterator lit=lKFs.begin(), lend= lKFs.end(); lit!=lend; lit++)
            {
                KeyFrame* pKFi=*lit;
                if(pKFi->mnRelocQuery!=F->mnId)
                {
                    pKFi->mnRelocWords=0; // 这一步操作是为了在下次遇到其他 Frame 时。清理原来存在的值！
                    pKFi->mnRelocQuery=F->mnId;
                    lKFsSharingWords.push_back(pKFi);
                }
                pKFi->mnRelocWords++;
            }
        }
    }
    if(lKFsSharingWords.empty())
        return vector<KeyFrame*>();

    // Only compare against those keyframes that share enough words.这里足够多的单词，是按照当前现有单词个数最多的百分比。制定阈值
    int maxCommonWords=0;
    // 找到与当前帧有最多共同单词的关键帧！
    for(list<KeyFrame*>::iterator lit=lKFsSharingWords.begin(), lend= lKFsSharingWords.end(); lit!=lend; lit++)
    {
        if((*lit)->mnRelocWords>maxCommonWords)
            maxCommonWords=(*lit)->mnRelocWords;
    }

    int minCommonWords = maxCommonWords*0.8f; // 80% 制定阈值

    list<pair<float,KeyFrame*> > lScoreAndMatch; // 存储符合上面 80% 阈值的关键帧，及其与当前帧相似性 BoW 得分 <score,KF*>

    int nscores=0;

    // Compute similarity score. 然后需要清理不符合要求的关键帧对应的 mRelocScore = 0;因为后面需要用,也就是后面 bug 的解决方式
    for(list<KeyFrame*>::iterator lit=lKFsSharingWords.begin(), lend= lKFsSharingWords.end(); lit!=lend; lit++)
    {
        KeyFrame* pKFi = *lit;

        if(pKFi->mnRelocWords>minCommonWords) // 只记录满足最小单词的得分
        {
            nscores++;
            float si = mpVoc->score(F->mBowVec,pKFi->mBowVec); // 计算两个词袋向量之间得分，计算相似性
            pKFi->mRelocScore=si;
            lScoreAndMatch.push_back(make_pair(si,pKFi));
        }
        else // 对于不满足最小共同单词的关键帧，需要做一下清理 mRelocScore = 0，因为这些帧很可能在上次重定位时计算了得分。但是此时却没有计算得分。
            // 为了不影响下面计算累计得分这里是下面的 bug 解决方案！
        {
            pKFi->mRelocScore = 0;
        }
    }

    if(lScoreAndMatch.empty()) // 实际上这里不会为真，前面已经统计过了最高单词个数。所以这里至少有一个
        return vector<KeyFrame*>();

    list<pair<float,KeyFrame*> > lAccScoreAndMatch; // 累计得分及其对应最高分的关键帧
    float bestAccScore = 0; // 在 lScoreAndMatch 中，每个关键帧都有一个临近关键帧组。计算临近关键帧组与当前帧的累计评分和。

    // Lets now accumulate score by covisibility 实际上，下面就是在这个变量 lScoreAndMatch 中选择关键帧集。因为该函数最上面已经记录了所有有共同单词的关键帧了。
            // 但是计算相似性评分的仅仅是符合最小单词个数的那些关键帧。下面的 accScore 实际能够累计的得分其实就是满足 lScoreAndMatch 中某几个关键帧得分的和。
    for(list<pair<float,KeyFrame*> >::iterator it=lScoreAndMatch.begin(), itend=lScoreAndMatch.end(); it!=itend; it++)
    {
        KeyFrame* pKFi = it->second;
        vector<KeyFrame*> vpNeighs = pKFi->GetBestCovisibilityKeyFrames(10); // 获取与当前帧有着最强共视关系的关键帧组

        float bestScore = it->first;
        float accScore = bestScore;
        KeyFrame* pBestKF = pKFi;
        for(vector<KeyFrame*>::iterator vit=vpNeighs.begin(), vend=vpNeighs.end(); vit!=vend; vit++)
        {
            KeyFrame* pKF2 = *vit;
            if(pKF2->mnRelocQuery!=F->mnId) // 下面 bug 解决方式，也可以是仿照上面哪个函数 DetectLoopCandidates()对应部分，在加上该条件 && pKF2->mnRelocWords<=minCommonWords
                continue;
            //bug处？？？？？：原因： 下面这里是与当前帧有共同单词的，但是可能没有计算相似性评分。
            accScore+=pKF2->mRelocScore; // 经过试验可以发现，在 pKFi 的临近关键帧集中，有些关键帧没有计算得分。那么此时这个 mRelocScore 变量内部值为垃圾值。
//            std::cout << "PKF2->mRelocScore: " << pKF2->mRelocScore << std::endl;
                      //<< "pKFi->mRelocScore: " << pKFi->mRelocScore << std::endl;
            if(pKF2->mRelocScore>bestScore) // 找最好得分及其关键帧
            {
                pBestKF=pKF2;
                bestScore = pKF2->mRelocScore;
            }

        }
        lAccScoreAndMatch.push_back(make_pair(accScore,pBestKF)); // 记录最好累计得分及其关键帧，下面需要在里面找合适的关键帧集（累计得分> 0.75*bestAccScore）
        if(accScore>bestAccScore) // 找到某个关键帧的共视图中最好的累计得分
            bestAccScore=accScore;
    }

    // Return all those keyframes with a score higher than 0.75*bestAccScore
    float minScoreToRetain = 0.75f*bestAccScore; // 累计得分阈值
    set<KeyFrame*> spAlreadyAddedKF; // 用来判断是否插入一个关键帧进来了
    vector<KeyFrame*> vpRelocCandidates; // 需要返回的。潜在的关键帧组！
    vpRelocCandidates.reserve(lAccScoreAndMatch.size());
    for(list<pair<float,KeyFrame*> >::iterator it=lAccScoreAndMatch.begin(), itend=lAccScoreAndMatch.end(); it!=itend; it++)
    {
        const float &si = it->first; // 累计得分
        if(si>minScoreToRetain)
        {
            KeyFrame* pKFi = it->second;
            if(!spAlreadyAddedKF.count(pKFi))
            {
                vpRelocCandidates.push_back(pKFi); // 潜在匹配的关键帧插入！
                spAlreadyAddedKF.insert(pKFi);
            }
        }
    }

    return vpRelocCandidates;
}

} //namespace ORB_SLAM
