
/* a hybrid recommender system for MovieLen */

import org.apache.spark.mllib.recommendation.ALS
import org.apache.spark.mllib.recommendation.MatrixFactorizationModel
import org.apache.spark.mllib.recommendation.Rating
import spark.implicits._

/*******************step 1: extract and tranform data ***********************/
//s_ratings.dat from MovieLens 100k
//users.dat from MovieLens 10M
//ratings.dat from MovieLens 10M

val movie_count = 3990

var users=sc.textFile("/FileStore/tables/qdmv8zm11488310977188/users.dat")
// movies.count()  = 17770
var raw_ratings=sc.textFile("/FileStore/tables/qdmv8zm11488310977188/ratings.dat")

// use small data to select model parameter. MovieLen has 1m 10m 20m
var small_raw_ratings= sc.textFile("/FileStore/tables/qdmv8zm11488310977188/s_ratings.dat")

// data format user::movie::rate::timestamp 
// put into Rating so that we can use ALS 


def convertRating(str:String): Rating = str.split("::") match {
    case Array(user,movie,rate,timestamp)=>Rating(user.toInt,movie.toInt,rate.toDouble)
}

case class User(var user:Int, var gender:Int,var age:Int,var occupation:Int) {
}

def convertUser(str:String) : User = {
  var strs=str.split("::")
  new User(strs(0).toInt,if(strs(1)=="M") 1 else 0,strs(2).toInt,strs(3).toInt)
}

var ratings= raw_ratings.map(convertRating).cache
var s_ratings= small_raw_ratings.map(convertRating).cache

/******************************** end of step 1 ************************************/



/*********************step 2: compute userprofile->score for new users ******************/ 

// here we build a userprofile->score map to deal with cold start
// score = 1*watched_times + 0.1*rating
// This formula is tunable to balance between popularity and quality.
// These days people are likely to watch bad but popular movie as social coins
// "Great Wall" eg. So we pick rating weight fairly low.

var usersDF= users.map(convertUser).toDF.repartition($"user").cache //for join speed
var ratingsDF = ratings.toDF

var userScores = 
usersDF.join(ratingsDF,usersDF.col("user")===ratingsDF.col("user"))
.select($"gender",$"age",$"occupation",$"product",$"rating")
.rdd
.map(row=>(row.getInt(0),row.getInt(1),row.getInt(2))->(row.getInt(3),row.getDouble(4)))
/* notice : naive implementation is not acceptable
 * it will shuffle array(movie_count) which will cause huge space/network stress
 * combineByKey( createCombiner, merger, merge combiner )
 * it will call create Combiner once for each key.
 * call merger on each partition. At last, combine all the accumulator from different partitions.
 * Knowing Internals helps improve efficiency
 */
.combineByKey(
  entry=>{
    // sparse vector in Spark doesn't support linear algebra. I see lots of people demand this.
    // Otherwise a sparse vector will perform much better.
  var scores = Array.fill(movie_count){0.0}
    scores(entry._1)= entry._2*0.1+1.0
    scores
  },
  (acc:Array[Double],entry)=>{
    acc(entry._1)+= entry._2*0.1+1.0
    acc
   }
  ,
  (acc1:Array[Double],acc2:Array[Double])=>{
    for( i <- 0 until movie_count){
      acc1(i)+= acc2(i)
    }
    acc1
  }
)

/**************************************end of step 2*******************************/



/*********************step 3: training small data to select paramters ******************************/ 

/* It is very hard to choose data that has similar stas property compared to other machine learning problem.
 * MoviewLens100k may not be a good representation of MoviewLens 10m
*/

 // compute rmse as the evaluation for model

def RMSE(model: MatrixFactorizationModel, data: RDD[Rating]): Double = {

    val predictions: RDD[Rating] = model.predict(data.map(r => (r.user, r.product)))
    val error = predictions.map{ r:Rating =>((r.user, r.product), r.rating)}
                                           .join(data.map{ r:Rating =>((r.user, r.product), r.rating)} )
                                           .values
                                           .map(r=>math.pow(r._1-r._2,2))  // use user and product as key, join two rating then find the error
  
    math.sqrt(error.mean())
}



 var splits = s_ratings.randomSplit(Array(0.8, 0.2))
 var training = splits(0).cache()
 var test = splits(1).cache()
    
val seed = 3L
val iterations = 20
//regularization_parameter 
val rps = List(0.01,0.05, 0.1, 0.3)
// ranks should consider datasize
val ranks = List(5, 10, 15, 20,25)


var min_error = Double.MaxValue
var best_rank = -1
var best_rp:Double = -1


/* Choosing standard: besides bias/variance error, it is important 
 * to note this rank cost memory k*user item*k needs to be stored
 * so pick a rank when increasing it won't give you much improvement.
 */


for (rank <- ranks){
  for(rp <- rps){
    // ALS algorithm. Note here we have explicit data instead implicit
  
    var model = ALS.train(training, rank, iterations, rp)
    var error = RMSE(model,test)
    println(error+","+rank+","+rp)
    if(error<min_error){
      min_error = error
      best_rank = rank
      best_rp = rp
    }
  }
}
  // for self implementation of ALS one can use normal regression
  // to minimize theta*x'. However, that is much less efficient 
  // in distributed environment because of two reasons: block to reduce shuffle,
  // compression representation to save memory: sparse vectors

println(s"best rank:$best_rank best regularization rate$best_rp")

//best rank:15 best regularization rate0.05 from our small dataset

/**************************************end of step 3*******************************/


/*********************step 4: train our model and make recomendation ******************************/ 

/*
 * Train the model. CF Recommendation suffers from cold start. New user and new movie is not prefered
 * In reality, people will build contend-based for fresh user/movie. 
 * But we don't have access to those data. User behavior movie feature etc.
 * But we do have some user profile data in MoviewLen1M, so using the previous matrix 
 * we build in step 2, we know if a new user fell into a particular type(gender,age,occupation)
 * we recommend them the top K for that group of people. If he/she rated more, we know more
 * about him/her. We move to the CF-based model.
 */
splits = ratings.randomSplit(Array(0.9, 0.1))
training = splits(0).cache()
test = splits(1).cache()

var model = ALS.train(training, best_rank, iterations, best_rp)
var error = RMSE(model,test)
//other than RMSE, learning curve and cost during gradient decent are all info that needs to be explored.


// recommend based on new user profile. As we don't have a running web, there is no
// new user. The following method is not tested. Just to illustrate idea.

def recommend(model: MatrixFactorizationModel, id:Int):Array[Int] ={
  val K = 10
  val waterMark = 5
  val count= usersDF.where($"user"===id).filter($"rating"===0).count
  val userProfile = usersDF.where($"user"===id).select($"gender",$"age",$"occupation").rdd.take(1)
  
  if (count < 5) {
    userScores.filter(e=>e._1===userProfile).collect
  }
  else
  {
    model.recommendProducts(id, K)
  }
  
}


/**************************************end of step 4*******************************/

/* model evaluation: 
best_rank: Int = 15 best_rp: Double = 0.05 
model: org.apache.spark.mllib.recommendation.MatrixFactorizationModel = org.apache.spark.mllib.recommendation.MatrixFactorizationModel@45620d6a 
error: Double = 0.8483929962240301
error is good back in 2010. Now there are deep learning algorithm minimizes this to 0.8
*/