private int scorerIndex(int n)
   {
      int lo = 0; // search starts array
      int hi = scorers.length - 1; // for first element less

      while (hi >= lo)
      {
         int mid = (lo + hi) >> 1;
         int midValue = starts[mid];
         if (n < midValue)
         {
            hi = mid - 1;
         }
         else if (n > midValue)
         {
            lo = mid + 1;
         }
         else
         { // found a match
            while (mid + 1 < scorers.length && starts[mid + 1] == midValue)
            {
               mid++; // scan to last match
            }
            return mid;
         }
      }
      return hi;
   }