# from .. import utils
from ._base import OpInfOperator

import numpy as np

import scipy.linalg as la
import itertools

# import scipy.sparse as sparse
from scipy.special import comb

__all__ = ["PolynomialOperator"]


class PolynomialOperator(OpInfOperator):
    r"""Polynomial state operator
    :math:`\Ophat_{\ell}(\qhat,\u) = \Phat[\qhat^{\otimes p}]`
    where :math:`\otimes p` indicates the Kronecker product of a vector with
    itself :math:`p` times. The matrix :math:`\Phat` is
    :math:`r \times \binom{r+p-1}{p}`.

    This class is equivalent to the following.

    * :math:`p = 1`: :class:`ConstantOperator`
    * :math:`p = 2`: :class:`QuadraticOperator`
    * :math:`p = 3`: :class:`CubicOperator`
    * :math:`p = 4`: :class:`QuarticOperator`

    Parameters
    ----------
    polynomial_order : int
        Order of the Kronecker product.
    entries : (r, (r + p - 1 choose p)) ndarray or None
        Operator matrix :math:`\Phat`.

    Examples
    --------
    >>> import numpy as np
    >>> H = opinf.operators.QuadraticOperator()
    >>> P = opinf.operators.PolynomialOperator(2)
    >>> entries = np.random.random((10, 100))
    >>> H.set_entries(entries)
    >>> H.shape
    (10, 55)
    >>> P.set_entries(H.entries)
    >>> q = np.random.random(10)
    >>> outH = H.apply(q)
    >>> out = P.apply(q)
    >>> np.allclose(out, outH)
    True
    """

    def __init__(self, polynomial_order: int, entries=None):
        """Initialize an empty operator."""
        if polynomial_order < 0 or (
            not np.isclose(polynomial_order, p := int(polynomial_order))
        ):
            raise ValueError(
                "expected non-negative integer polynomial order"
                + f" polynomial_order. Got p={polynomial_order}"
            )

        self.polynomial_order = p

        super().__init__(entries=entries)

    def copy(self):
        """Return a copy of the operator."""
        entries = self.entries.copy() if self.entries is not None else None
        return self.__class__(
            entries=entries, polynomial_order=self.polynomial_order
        )

    def operator_dimension(self, r: int, m=None) -> int:
        """
        computes the number of non-redundant terms in a vector of length r
        that is taken to the power p with the Kronecker product

        Parameters
        ----------
        r : int
            State dimension.
        m : int or None
            Input dimension -- currently not used
        """
        if r < 0 or (not np.isclose(r, int(r))):
            raise ValueError(
                f"expected non-negative integer reduced dimension r. Got r={r}"
            )
        return PolynomialOperator.polynomial_operator_dimension(
            r=r, polynomial_order=self.polynomial_order
        )

    @staticmethod
    def polynomial_operator_dimension(r, polynomial_order) -> int:
        if polynomial_order == 0:
            return 1
        return comb(r, polynomial_order, repetition=True, exact=True)

    def datablock(self, states: np.ndarray, inputs=None) -> np.ndarray:
        r"""Return the data matrix block corresponding to
        this operator's polynomial order,
        with ``states`` being the projected snapshots.

        Parameters
        ----------
        states : (r, k) or (k,) ndarray
            State vectors. Each column is a single state vector.
            If one dimensional, it is assumed that :math:`r = 1`.
        inputs : (m, k) or (k,) ndarray or None
            Input vectors (not used).

        Returns
        -------
        datablock : (self.operator_dimension(r), k) ndarray
            where p is the polynomial order for this operator.
        """
        # if constant, we just return an array containing ones
        # of shape 1 x <number of data points>
        if self.polynomial_order == 0:
            return np.ones((1, np.atleast_1d(states).shape[-1]))

        # make sure data is in 2D
        states = np.atleast_2d(states)

        if states.shape[0] == 0:
            return np.empty(shape=(0, states.shape[1]))

        # compute data matrix
        return PolynomialOperator.exp_p(states, self.polynomial_order)

    @staticmethod
    def keptIndices_p(r, p):
        """
        returns the non-redundant indices in a kronecker-product with
        exponent p when the dimension of the vector is r
        """
        if p == 0:
            return np.array([0])

        dim_if_p_was_one_smaller = PolynomialOperator(
            polynomial_order=(p - 1)
        ).operator_dimension(r=r)
        indexmatrix = np.reshape(
            np.arange(r * dim_if_p_was_one_smaller),
            (r, dim_if_p_was_one_smaller),
        )
        return np.hstack(
            [
                indexmatrix[
                    i,
                    : PolynomialOperator(
                        polynomial_order=(p - 1)
                    ).operator_dimension(i + 1),
                ]
                for i in range(r)
            ]
        )

    @staticmethod
    def exp_p(x, p, kept=None):
        """
        recursively computes x^p without the redundant terms
        (it still computes them but then takes them out)
        the result has shape

        if x is 1-dimensional:
        (PolynomialOperator.operator_dimension(x.shape[0]),)

        otherwise:
        (PolynomialOperator.operator_dimension(x.shape[0]), x.shape[1])
        """
        # for a constant operator, we just return 1 (x^0 = 1)
        if p == 0:
            return np.ones([1])

        # for a linear operator, x^1 = 1
        if p == 1:
            return x

        # identify kept entries in condensed Kronecker product for
        # this reduced dimension
        # for all polynomial orders up to self.polynomial order
        if kept is None:
            r = x.shape[0]
            kept = [
                PolynomialOperator.keptIndices_p(r=r, p=i)
                for i in range(p + 1)
            ]

        # distinguish between the shapes of the input
        if len(x.shape) == 1:
            # this gets called when the ROM is run
            return np.kron(x, PolynomialOperator.exp_p(x, p - 1, kept))[
                kept[p]
            ]
        else:
            # this gets called for constructing the data matrix
            return la.khatri_rao(x, PolynomialOperator.exp_p(x, p - 1, kept))[
                kept[p]
            ]

    @staticmethod
    def ckron_indices(r, p):
        """Construct a mask for efficiently computing the compressed, p-times
        repeated Kronecker product.

        This method provides a faster way to evaluate :meth:`ckron`
        when the state dimension ``r`` is known *a priori*.

        Parameters
        ----------
        r : int
            State dimension.
        p: int
            polynomial order

        Returns
        -------
        mask : ndarray
            Compressed Kronecker product mask.

        """
        mask = np.array(
            [*itertools.combinations_with_replacement([*range(r)][::-1], p)]
        )[::-1, :]
        return mask

    def apply(self, state: np.ndarray, input_=None) -> np.ndarray:
        r"""Apply the operator to the given state. Input is not used.
        See OpInfOperator.apply for description.
        """
        # note: having all these if conditions here to distinguish between
        # p=0, p=1, p>1
        # really makes things slower than necessary (apply gets called a lot)
        # it might be worth excluding these special cases from the
        # PolynomialOperator
        # class and defaulting to ConstantOperator and LinearOperator instead.

        if state.shape[0] != self.state_dimension:
            raise ValueError(
                f"Expected state of dimension r={self.state_dimension}."
                + f"Got state.shape={state.shape}"
            )

        # constant
        if self.polynomial_order == 0:
            if np.ndim(state) == 2:  # r, k > 1.
                return np.outer(self.entries, np.ones(state.shape[-1]))
            if np.ndim(self.entries) == 2:
                return self.entries[:, 0]
            return self.entries
        # note: no need to go through the trouble of identifying the
        # non-redundant indices

        # linear
        if self.polynomial_order == 1:
            return self.entries @ state
        # note: no need to go through the trouble of identifying the
        # non-redundant indices

        restricted_kronecker_product = np.prod(
            state[self.nonredudant_entries_mask], axis=1
        )
        return self.entries @ restricted_kronecker_product

    # Properties --------------------------------------------------------------
    @property
    def nonredudant_entries(self) -> list:
        r"""list containing at index i a list of the indices that are kept
        when restricting the i-times Kronecker product of a vector of
        shape self.state_dimension() with itself.
        """
        return [
            PolynomialOperator.keptIndices_p(r=self.state_dimension, p=i)
            for i in range(self.polynomial_order + 1)
        ]

    @property
    def nonredudant_entries_mask(self) -> np.ndarray:
        r"""list containing at index i a list of the indices that are kept
        when restricting the i-times Kronecker product of a vector of
        shape self.state_dimension() with itself.
        """
        return self.ckron_indices(
            r=self.state_dimension, p=self.polynomial_order
        )

    def restrict_to_subspace(self, indices_trial, indices_test=None):
        r"""
        Creates a new operator of type `PolynomialOperator` of the same
        polynomial order as this one but for the reduced (trial) dimension
        :math:`r_1 :=` ``len(indices)`` and (test) dimension
        ``len(indices_test)`` (Petrov-Galerkin setting). The new operator
        is constructed by restricting the action of this operator (``self``)
        onto :math:`span{\mathbf{v}_i: i \in indices_trial}`, and testing
        in :math:`span{\mathbf{v}_i: i \in indices_test}`.

        If ``indices_test``
        is not provided, defaults to the Galerkin setting
        ``indices_test = indices_trial``.

        Currently, the more general restriction onto combinations of
        basis vectors (e.g., onto :math:`span{(v_1+v_2)/2}`) is not supported.

        Parameters
        ----------
        indices_trial : list of integers
            indices of the (trial) basis vectors onto which the operator
            shall be restricted. Needs to be in increasing order and
            not contain dubplicates.
        indices_test : list of integers
            indices of the (test) basis vectors onto which the operator
            shall be restricted in the Petrov-Galerkin setting in
            increasing order. Needs to be in increasing order and
            not contain dubplicates.

        Returns
        -------
        PolynomialOperator
            Operator for trial dimension ``len(indices_trial)``, test
            dimension ``len(indices_test)``, and polynomial order
            ``self.polynomial_order``.
        """
        if indices_test is None:
            indices_test = indices_trial

        if max(indices_trial) >= self.state_dimension:
            raise RuntimeError(
                f"""
                               In PolynomialOperator.restrict_to_subspace:
                               Encountered restriction onto unknown trial basis
                               vector number {max(indices_trial)}.
                               Reduced dimension is {self.state_dimension}"""
            )

        if max(indices_test) >= self.state_dimension:
            raise RuntimeError(
                f"""
                               In PolynomialOperator.restrict_to_subspace:
                               Encountered restriction onto unknown test basis
                               vector number {max(indices_test)}.
                               Reduced dimension is {self.state_dimension}"""
            )

        new_entries = PolynomialOperator._restrict_matrix_to_subspace(
            indices_trial=indices_trial,
            entries=self.entries,
            polynomial_order=self.polynomial_order,
            indices_test=indices_test,
        )

        return PolynomialOperator(
            entries=new_entries, polynomial_order=self.polynomial_order
        )

    @staticmethod
    def _restrict_matrix_to_subspace(
        indices_trial, entries, polynomial_order, indices_test=None
    ):
        r"""
        Treating the matrix `entries` as operator matrix for the
        polynomial order `polynomial_order`, this function creates
        a submatrix `entries_sub` by restricting `entries` onto
        those columns that correspond to interactions of basis
        vectors :math:`v_i` with :math:`i \in` ``indices_trial``
        and to the rows listed in `indices_test`.

        Defaults to ``indices_test=indices_trial`` if
        ``indices_test = None``.

        Parameters
        ----------
        indices_trial : list of integers
            indices of the (trial) basis vectors onto which the operator
            shall be restricted. Needs to be in increasing order and
            not contain dubplicates.
        entries : np.ndarray
            operator entry matrix of shape :math:`(a,b)` with
            :math:`a \ge ` ``len(indices_test)`` and
            :math:`b \ge r^{(p)}` where :math:`r=` ``len(indices_trial)``
            and :math:`r^{(p)}` is the number of non-redundant entries for
            a polynomial operator of order :math:`p` and dimension :math:`r`.
        polynomial_order : int
            polynomial order of the operator matrix to be extracted
        indices_test : list of integers
            indices of the (test) basis vectors onto which the operator
            shall be restricted in the Petrov-Galerkin setting in
            increasing order. Needs to be in increasing order and
            not contain dubplicates.

        Returns
        -------
        entries_sub : np.ndarray
            of shape ``(len(indices_test), c)``, where `c` is the
            number of non-redundant entries for the condensed
            Kronecker product of a vector with dimension
            ``len(indices_trial)``.
        """
        if len(indices_trial) != len(set(indices_trial)):
            raise RuntimeError(
                f"""
                In PolynomialOperator.restrict_matrix_to_subspace:
                Received duplicate entries in
                `indices_trial=`{indices_trial}"""
            )

        if indices_test is None:
            indices_test = indices_trial
        elif len(indices_test) != len(set(indices_test)):
            raise RuntimeError(
                f"""
                In PolynomialOperator.restrict_matrix_to_subspace:
                Received duplicate entries in
                `indices_test=`{indices_test}"""
            )

        if indices_trial != sorted(indices_trial) or indices_test != sorted(
            indices_test
        ):
            raise RuntimeError(
                f"""
        In PolynomialOperator._extend_matrix_to_dimension: Received unordered
        indices {indices_trial} (trial) or {indices_test} (test).
        """
            )

        # constant
        if polynomial_order == 0:
            if np.ndim(entries) == 2:
                return entries[indices_test, 0]
            return entries[indices_test]

        # higher-order polynomials
        entries_sub = entries[indices_test, :]
        # restrict to test indices only

        col_indices = PolynomialOperator._columnIndices_p(
            indices=indices_trial, p=polynomial_order
        )
        # find out which columns to keep

        return entries_sub[:, col_indices]

    @staticmethod
    def _columnIndices_p(indices, p):
        r"""
        Identifies all (column) indices of a polynomial operator
        of polynomial order :math:`p` that encode interactions
        of basis vectors :math:`v_i` with :math:`i\in` ``indices``.

        Parameters
        ----------
        indices : list of integers
            indices of the basis vectors for which interactions
            shall be identified
        p : int
            polynomial order of the interactions
        """
        if p == 1:
            return indices

        if p == 0:
            return [0]

        sub = PolynomialOperator._columnIndices_p(indices, p - 1)
        return [
            comb(indices[i], p, repetition=True, exact=True) + sub[j]
            for i in range(len(indices))
            for j in range(
                PolynomialOperator.polynomial_operator_dimension(
                    r=i + 1, polynomial_order=p - 1
                )
            )
        ]

    def extend_to_dimension(
        self, new_r, indices_trial=None, indices_test=None, new_r_test=None
    ):
        r"""
        Creates a new operator of type `PolynomialOperator` of the same
        polynomial order as this one but for the reduced (trial) dimension
        ``new_r`` and (test) dimension ``new_r_test`` (defaulted to
        ``new_r_test = new_r`` if not provided). The new operator is
        created by mapping the current (trial) basis vector
        :math:`\mathbf{v}_i`
        onto the basis vector :math:`\tilde{\mathbf{v}}_j`,
        :math:`j=` ``indices_trial[i]`` of the new basis, :math`i=1, ..., r`.
        Similarly, the current test basis vectors :math:`\mathbf{w}_i` are
        mapped onto the new test vectors :math:`\tilde{\mathbf{w}}_j`,
        :math:`j=` ``indices_test[i]`` of the new basis, :math`i=1, ..., r`.
        The remaining actions of the new operator (i.e., all actions that
        involve :math:`\tilde{\mathbf{v}}_j` with
        :math:`j\notin` ``indices_trial`` or :math:`\tilde{\mathbf{w}}_j`
        with :math:`j\notin` ``indices_test``) are defaulted to 0.

        If ``indices_trial`` is not provided, it is assumed that the
        current basis is expanded and the current basis vectors are
        to be mapped onto the first :math:`r` basis vectors of the
        new basis, i.e., we default to ``indices_trial = [0, ..., r-1]``.

        If ``indices_test``
        is not provided, defaults to the Galerkin setting
        ``indices_test = indices_trial``.

        Currently, the more general restriction onto combinations of
        basis vectors (e.g., onto :math:`span{(v_1+v_2)/2}`) is not supported.

        Parameters
        ----------
        new_r : int
            target reduced dimension (trial space). Needs to be at
            least as large as ``self.state_dimension``
        indices_trial : list of integers
            indices of the (trial) basis vectors to which the previous
            operator entries shall be mapped in the expanded basis.
            Needs to be in increasing order and
            not contain dubplicates.
        indices_test : list of integers
            indices of the (test) basis vectors onto which the
            previous operator entries shall be mapped in the
            expanded basis (Petrov-Galerkin setting only).
            Needs to be in increasing order and
            not contain dubplicates.
        new_r_test : int
            target reduced dimension (test space). Defaulted to
            ``new_r`` if not provided.

        Returns
        -------
        PolynomialOperator
            Operator for trial dimension ``new_r``, test
            dimension ``new_r_test``, and polynomial order
            ``self.polynomial_order``.
        """
        if indices_trial is None:
            # default to extending the basis towards the right
            indices_trial = [*range(self.state_dimension)]

        if indices_test is None:
            # default to Galerking case
            indices_test = indices_trial

        if new_r < self.state_dimension:
            raise RuntimeError(
                f"""In PolynomialOperator.extend_to_dimension:
                Dimension mismatch. Expected new dimension ({new_r})
                to be larger than old dimension ({self.state_dimension})
                """
            )

        new_entries = PolynomialOperator._extend_matrix_to_dimension(
            new_r=new_r,
            indices_trial=indices_trial,
            polynomial_order=self.polynomial_order,
            old_entries=self.entries,
            indices_test=indices_test,
            new_r_test=new_r_test,
        )

        return PolynomialOperator(
            polynomial_order=self.polynomial_order, entries=new_entries
        )

    @staticmethod
    def _extend_matrix_to_dimension(
        new_r,
        indices_trial,
        polynomial_order,
        old_entries,
        indices_test=None,
        new_r_test=None,
    ):
        r"""
        This is the reverse function to _restrict_marix_to_dimension.

        Treating the matrix ``old_entries`` as operator matrix for the
        polynomial order ``polynomial_order``, this function creates
        a larger matrix of shape ``(new_r_test, a)`` with
        :math:`a=r_{new}^{(p)}` the number of non-redundant entries
        in the :math:`p=` ``polynomial-oder``-fold Kronecker product
        of an :math:`r_{new}=` ``new_r`` dimensional vector.
        The new matrix constains ``old_entries`` as the submatrix
        encoding operator actions for trial and test basis vectors
        with indices in ``indices_trial`` and ``indices_test``.
        All remaining entries are set to zero.

        Defaults to ``indices_test=indices_trial`` if
        ``indices_test = None``.

        Parameters
        ----------
        new_r : int
            target reduced dimension (trial space). Needs to be at
            least as large as ``self.state_dimension``
        indices_trial : list of integers
            indices of the (trial) basis vectors to which the previous
            operator entries shall be mapped in the expanded basis.
            Needs to be in increasing order and
            not contain dubplicates.
        polynomial_order : int
            polynomial order of the operator matrix to be extracted
        old_entries : np.ndarray
            operator entry matrix of shape ``(a, b)`` with
            :math:`a =` ``len(indices_test)`` and
            :math:`b \ge r^{(p)}` where :math:`r=` ``len(indices_trial)``
            and :math:`r^{(p)}` is the number of non-redundant entries for
            a polynomial operator of order :math:`p` and dimension :math:`r`.
        indices_test : list of integers
            indices of the (test) basis vectors onto which the
            previous operator entries shall be mapped in the
            expanded basis (Petrov-Galerkin setting only).
            Needs to be in increasing order and
            not contain dubplicates.
        new_r_test : int
            target reduced dimension (test space). Defaulted to
            ``new_r`` if not provided.

        Returns
        -------
        new_matrix : np.ndarray
            of shape ``(new_r_test, c)``, where `c` is the
            number of non-redundant entries for the condensed
            Kronecker product of a vector with dimension
            ``new_r`` for polynomial order ``polynomial_order``.
            Contains `old_entries` as submatrix.
        """
        if indices_test is None:
            indices_test = indices_trial

        if new_r_test is None:
            new_r_test = new_r

        if indices_trial != sorted(indices_trial) or indices_test != sorted(
            indices_test
        ):
            raise RuntimeError(
                f"""
                In PolynomialOperator._extend_matrix_to_dimension:
                Received unordered indices
                {indices_trial} (trial) or {indices_test} (test).
                """
            )

        old_r = len(indices_trial)
        if not old_entries.shape == (
            len(indices_test),
            PolynomialOperator.polynomial_operator_dimension(
                r=old_r, polynomial_order=polynomial_order
            ),
        ):
            raise RuntimeError(
                f"""In PolynomialOperator._extend_matrix_to_dimension:
                Mismatch in the dimension of the passed matrix.
                Expected {
                    (len(indices_test),
                     PolynomialOperator.polynomial_operator_dimension(r=old_r,
                     polynomial_order=polynomial_order))}.
                Got {old_entries.shape}.
                """
            )

        if old_r > new_r:
            raise RuntimeError(
                f"""In PolynomialOperator.
                               _extend_matrix_to_dimension:
                               Mismatch in passed indices:
                               Old dimension {old_r} is larger
                               than new dimension {new_r}
                               """
            )

        if len(indices_test) > new_r_test:
            raise RuntimeError(
                f"""In PolynomialOperator.
                _extend_matrix_to_dimension:
                               Mismatch in passed indices for test space:
                               Old dimension {len(indices_test)} is
                               larger than new test dimension {new_r_test}
                               """
            )

        # initialize matrix for old test space dimension
        new_marix_sub = np.zeros(
            shape=(
                len(indices_test),
                PolynomialOperator.polynomial_operator_dimension(
                    r=new_r, polynomial_order=polynomial_order
                ),
            )
        )

        # populate columns
        col_indices = PolynomialOperator._columnIndices_p(
            indices=indices_trial, p=polynomial_order
        )
        new_marix_sub[:, col_indices] = old_entries

        # final matrix: fill remaining rows with zeros
        new_matrix = np.zeros(
            shape=(
                new_r_test,
                PolynomialOperator.polynomial_operator_dimension(
                    r=new_r, polynomial_order=polynomial_order
                ),
            )
        )
        new_matrix[indices_test, :] = new_marix_sub

        return new_matrix
