public double normalizedDistance(double[] point1, double[] point2) throws Exception {
    return Math.sqrt(distance2(point1, point2)) / point1.length;
  }