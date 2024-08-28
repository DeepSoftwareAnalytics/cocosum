@Override
    boolean feasible(int n, int m) {

        // verify atom semantic feasibility
        if (!atomMatcher.matches(container1.getAtom(n), container2.getAtom(m))) return false;

        // unmapped terminal vertices n and m are adjacent to
        int nTerminal1 = 0, nTerminal2 = 0;
        // unmapped non-terminal (remaining) vertices n and m are adjacent to
        int nRemain1 = 0, nRemain2 = 0;

        // 0-look-ahead: check each adjacent edge for being mapped, and count
        // terminal or remaining
        for (int n_prime : g1[n]) {
            int m_prime = m1[n_prime];

            // v is already mapped, there should be an edge {m, w} in g2.
            if (m_prime != UNMAPPED) {
                IBond bond2 = bonds2.get(m, m_prime);
                if (bond2 == null) // the bond is not present in the target
                    return false;
                // verify bond semantic feasibility
                if (!bondMatcher.matches(bonds1.get(n, n_prime), bond2)) return false;
            } else {
                if (t1[n_prime] > 0)
                    nTerminal1++;
                else
                    nRemain1++;
            }
        }

        // monomorphism: each mapped edge in g2 doesn't need to be in g1 so
        // only the terminal and remaining edges are counted
        for (int m_prime : g2[m]) {
            if (m2[m_prime] == UNMAPPED) {
                if (t2[m_prime] > 0)
                    nTerminal2++;
                else
                    nRemain2++;
            }
        }

        // 1-look-ahead : the mapping {n, m} is feasible iff the number of
        // terminal vertices (t1) adjacent to n is less than or equal to the
        // number of terminal vertices (t2) adjacent to m.
        //
        // 2-look-ahead: the mapping {n, m} is feasible iff the number of
        // vertices adjacent to n that are neither in m1 or t1 is less than or
        // equal to the number of the number of vertices adjacent to m that
        // are neither in m2 or t2. To allow mapping of monomorphisms we add the
        // number of adjacent terminal vertices.
        return nTerminal1 <= nTerminal2 && (nRemain1 + nTerminal1) <= (nRemain2 + nTerminal2);
    }