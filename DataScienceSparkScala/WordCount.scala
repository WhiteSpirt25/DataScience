import org.apache.spark.{SparkConf, SparkContext}

object WordCount {
  def main(args: Array[String]) {

    //Create a SparkContext to initialize Spark
    val conf = new SparkConf()
    conf.setMaster("local[4]")
    conf.setAppName("Word Count")
    val sc = new SparkContext(conf)

    //reading text file
    val textFile = sc.textFile("text.txt")
    val counts = textFile.flatMap(line => line.split(" ")).map(word => (word,1)).reduceByKey(_ + _)
    counts.collect().foreach(println)
    //reading list of required words
    val reqFile = sc.textFile("required.txt")
    val req = reqFile.flatMap(line => line.split(" ")).map(word => (word,1)).reduceByKey(_ + _)
    req.collect().foreach(println)
    println("Answer")
    val cross = counts.join(req).map(x => (x._1 , x._2._1))
    cross.collect().foreach(println)
  }
}
