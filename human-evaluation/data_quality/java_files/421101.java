public static int dfs(int j, int k, int[] Pinv, int[] Llen, int Llen_offset,
			int[] Lip, int Lip_offset,
			int[] Stack, int[] Flag, int[] Lpend, int top, double[] LU,
			double[] Lik, int Lik_offset, int[] plength, int[] Ap_pos)
	{
		int i, pos, jnew, head, l_length;
		/*int[]*/double[] Li;

		l_length = plength [0] ;

		head = 0 ;
		Stack [0] = j ;
		ASSERT (Flag [j] != k) ;

		while (head >= 0)
		{
			j = Stack [head] ;
			jnew = Pinv [j] ;
			ASSERT (jnew >= 0 && jnew < k) ;        /* j is pivotal */

			if (Flag [j] != k)          /* a node is not yet visited */
			{
				/* first time that j has been visited */
				Flag [j] = k ;
				PRINTF ("[ start dfs at %d : new %d\n", j, jnew) ;
				/* set Ap_pos [head] to one past the last entry in col j to scan */
				Ap_pos [head] =
					(Lpend [jnew] == EMPTY) ?  Llen [Llen_offset + jnew] : Lpend [jnew] ;
			}

			/* add the adjacent nodes to the recursive stack by iterating through
			 * until finding another non-visited pivotal node */
			Li = LU ;
			int Li_offset = Lip [Lip_offset + jnew] ;
			for (pos = --Ap_pos [head] ; pos >= 0 ; --pos)
			{
				i = (int) Li [Li_offset + pos] ;
				if (Flag [i] != k)
				{
					/* node i is not yet visited */
					if (Pinv [i] >= 0)
					{
						/* keep track of where we left off in the scan of the
						 * adjacency list of node j so we can restart j where we
						 * left off. */
						Ap_pos [head] = pos ;

						/* node i is pivotal; push it onto the recursive stack
						 * and immediately break so we can recurse on node i. */
						Stack [++head] = i ;
						break ;
					}
					else
					{
						/* node i is not pivotal (no outgoing edges). */
						/* Flag as visited and store directly into L,
						 * and continue with current node j. */
						Flag [i] = k ;
						Lik [Lik_offset + l_length] = i ;
						l_length++ ;
					}
				}
			}

			if (pos == -1)
			{
				/* if all adjacent nodes of j are already visited, pop j from
				 * recursive stack and push j onto output stack */
				head-- ;
				Stack[--top] = j ;
				PRINTF ("  end   dfs at %d ] head : %d\n", j, head) ;
			}
		}

		plength[0] = l_length ;
		return (top) ;
	}