from __future__ import absolute_import
import operator
import collections
import numpy as np
import itertools
import functools
import firedrake
from coffee import base as ast
from coffee.visitor import Visitor

import ufl
from ufl.algorithms.map_integrands import map_integrand_dags
from ufl.algorithms.multifunction import MultiFunction

firedrake.parameters["coffee"] = {}

firedrake.parameters["pyop2_options"]["debug"] = True

np.set_printoptions(linewidth=200, precision=4)


class CheckRestrictions(MultiFunction):
    expr = MultiFunction.reuse_if_untouched

    def negative_restricted(self, o):
        raise ValueError("Cell-wise integrals may only contain positive restrictions")


class RemoveRestrictions(MultiFunction):
    expr = MultiFunction.reuse_if_untouched

    def positive_restricted(self, o):
        return self(o.ufl_operands[0])


class Tensor(object):
    id = 0

    def __init__(self, arguments, coefficients):
        self.id = Tensor.id
        Tensor.id += 1
        self._arguments = tuple(arguments)
        self._coefficients = tuple(coefficients)
        shape = []
        shapes = {}
        for i, arg in enumerate(self._arguments):
            V = arg.function_space()
            shp = []
            for fs in V:
                shp.append(fs.fiat_element.space_dimension() * fs.cdim)
            shapes[i] = tuple(shp)
            shape.append(sum(shp))
        self.shapes = shapes
        self.shape = tuple(shape)

    def arguments(self):
        return self._arguments

    def coefficients(self):
        return self._coefficients

    @property
    def T(self):
        return Transpose(self)

    @property
    def inv(self):
        return Inverse(self)

    @classmethod
    def check_integrals(cls, integrals):
        mapper = CheckRestrictions()
        for it in integrals:
            map_integrand_dags(mapper, it)

    def _bop(self, other, op):
        d = {operator.add: Sum,
             operator.sub: Sub,
             operator.mul: Mul}
        assert isinstance(other, Tensor)
        return d[op](self, other)

    @property
    def operands(self):
        return ()

    def __str__(self, prec=None):
        return "FNORD"

    def __add__(self, other):
        return self._bop(other, operator.add)

    def __sub__(self, other):
        return self._bop(other, operator.sub)

    def __mul__(self, other):
        return self._bop(other, operator.mul)

    def __neg__(self):
        return Neg(self)

    def __pos__(self):
        return Pos(self)


class Scalar(Tensor):
    def __init__(self, form):
        assert len(form.arguments()) == 0
        self.check_integrals(form.integrals())
        self.form = form
        Tensor.__init__(self, arguments=(),
                        coefficients=form.coefficients())


class Vector(Tensor):
    def __init__(self, form):
        assert len(form.arguments()) == 1
        self.check_integrals(form.integrals())
        self.form = form
        Tensor.__init__(self, arguments=form.arguments(),
                        coefficients=form.coefficients())

    def __str__(self, prec=None):
        return "V_%d" % self.id

    __repr__ = __str__


class Matrix(Tensor):
    def __init__(self, form):
        assert len(form.arguments()) == 2
        self.check_integrals(form.integrals())
        self.form = form
        Tensor.__init__(self, arguments=form.arguments(),
                        coefficients=form.coefficients())

    def __str__(self, prec=None):
        return "M_%d" % self.id

    __repr__ = __str__


class Inverse(Tensor):
    def __init__(self, tensor):
        assert len(tensor.shape) == 2 and tensor.shape[0] == tensor.shape[1], \
            "Inverse only makes sense for square tensors"
        self.tensor = tensor
        Tensor.__init__(self, arguments=reversed(tensor.arguments()),
                        coefficients=tensor.coefficients())

    @property
    def operands(self):
        return (self.tensor, )

    def __str__(self, prec=None):
        return "%s.inverse()" % self.tensor

    def __repr__(self):
        return "Inverse(%s)" % self.tensor


class Transpose(Tensor):
    def __init__(self, tensor):
        self.tensor = tensor
        Tensor.__init__(self, arguments=reversed(tensor.arguments()),
                        coefficients=tensor.coefficients())

    @property
    def operands(self):
        return (self.tensor, )

    def __str__(self, prec=None):
        return "%s.transpose()" % self.tensor

    def __repr__(self):
        return "Transpose(%r)" % self.tensor


class UOp(Tensor):
    def __init__(self, tensor):
        self.tensor = tensor
        Tensor.__init__(self, arguments=tensor.arguments(),
                        coefficients=tensor.coefficients())

    @property
    def operands(self):
        return (self.tensor, )

    def __str__(self, prec=None):
        d = {operator.neg: '-',
             operator.pos: '+'}
        if prec is None or self.prec >= prec:
            par = lambda x: x
        else:
            par = lambda x: "(%s)" % x

        return par("%s%s" % (d[self.op], self.tensor.__str__(prec=self.prec)))

    def __repr__(self):
        return "%s(%r)" % (type(self).__name__, self.tensor)


class Neg(UOp):
    op = operator.neg
    prec = 1


class Pos(UOp):
    op = operator.pos
    prec = 1


class BinOp(Tensor):
    def __init__(self, left, right):
        args = self.get_arguments(left, right)
        coeffs = self.get_coefficients(left, right)
        self.left = left
        self.right = right
        Tensor.__init__(self, arguments=args,
                        coefficients=coeffs)

    @property
    def operands(self):
        return (self.left, self.right)

    @classmethod
    def get_arguments(cls, a, b):
        pass

    @classmethod
    def get_coefficients(cls, a, b):
        # Merge dropping dups from b
        coeffs = []
        got = set(a.coefficients())
        for c in b.coefficients():
            if c not in got:
                coeffs.append(c)
        return tuple(list(a.coefficients()) + coeffs)

    def __str__(self, prec=None):
        d = {operator.add: '+',
             operator.sub: '-',
             operator.mul: '*'}
        if prec is None or self.prec >= prec:
            par = lambda x: x
        else:
            par = lambda x: "(%s)" % x

        left = self.left.__str__(prec=self.prec)
        right = self.right.__str__(prec=self.prec)
        val = "%s %s %s" % (left, d[self.op], right)
        return par(val)

    def __repr__(self):
        return "%s(%r, %r)" % (type(self).__name__, self.left, self.right)


class Sum(BinOp):
    prec = 1
    op = operator.add

    @classmethod
    def get_arguments(cls, a, b):
        # Scalars distribute
        if isinstance(a, Scalar):
            return b.arguments()
        elif isinstance(b, Scalar):
            return a.arguments()
        assert a.shape == b.shape
        return a.arguments()


class Sub(BinOp):
    prec = 1
    op = operator.sub

    @classmethod
    def get_arguments(cls, a, b):
        # Scalars distribute
        if isinstance(a, Scalar):
            return b.arguments()
        elif isinstance(b, Scalar):
            return a.arguments()
        assert a.shape == b.shape
        return a.arguments()


class Mul(BinOp):
    prec = 2
    op = operator.mul

    @classmethod
    def get_arguments(cls, a, b):
        # Scalars distribute
        if isinstance(a, Scalar):
            return b.arguments()
        elif isinstance(b, Scalar):
            return a.arguments()
        # Contraction over middle indices
        assert a.arguments()[-1].function_space() == b.arguments()[0].function_space()
        return a.arguments()[:-1] + b.arguments()[1:]


class TransformKernelTensor(Visitor):
    """Replace all references to an output tensor with specified name
    by an Eigen Matrix of the appropriate shape.

    A default name of :data:`"A"` is assumed, otherwise you can pass
    in the name as the :data:`name` keyword argument when calling the
    visitor.
    """
    def visit_object(self, o, *args, **kwargs):
        return o

    def visit_list(self, o, *args, **kwargs):
        new = [self.visit(e, *args, **kwargs) for e in o]
        if all(n is e for n, e in zip(new, o)):
            return o
        return new

    visit_Node = Visitor.maybe_reconstruct

    def visit_FunDecl(self, o, *args, **kwargs):
        new = self.visit_Node(o, *args, **kwargs)
        ops, okwargs = new.operands()
        name = kwargs.get("name", "A")
        if all(new is old for new, old in zip(ops, o.operands()[0])):
            return o
        pred = ["template<typename Derived>", "static", "inline"]
        ops[4] = pred
        body = ops[3]
        args, _ = body.operands()
        nargs = [ast.FlatBlock("Eigen::MatrixBase<Derived>& %s = const_cast< Eigen::MatrixBase<Derived>& >(%s_);\n" % \
                               (name, name))] + args
        ops[3] = ast.Node(nargs)
        return o.reconstruct(*ops, **okwargs)

    def visit_Decl(self, o, *args, **kwargs):
        # Found tensor
        name = kwargs.get("name", "A")
        if o.sym.symbol != name:
            return o
        typ = "Eigen::MatrixBase<Derived> const &"
        return o.reconstruct(typ, ast.Symbol("%s_" % name))

    def visit_Symbol(self, o, *args, **kwargs):
        name = kwargs.get("name", "A")
        if o.symbol != name:
            return o
        shape = o.rank
        return ast.FunCall(ast.Symbol(name), *shape)


def get_kernel(expr):
    dtype = "double"
    shape = expr.shape
    syms = {}
    kernels = {}

    needs_cell_facets = False
    def matrix_type(shape):
        if len(shape) == 1:
            rows = shape[0]
            cols = 1
        else:
            assert len(shape) == 2, "%d-rank tensors unsupported" % len(shape)
            rows = shape[0]
            cols = shape[1]
        if cols != 1:
            order = ", Eigen::RowMajor"
        else:
            order = ""
        return "Eigen::Matrix<double, %d, %d%s>" % (rows, cols, order)

    def map_type(matrix):
        return "Eigen::Map<%s >" % matrix

    statements = []
    def get_decls(expr):
        if isinstance(expr, (Scalar, Vector, Matrix)):
            if syms.get(expr):
                return
            sym = ast.Symbol("sym%d" % len(syms))
            typ = matrix_type(expr.shape)

            syms[expr] = sym
            statements.append(ast.Decl(typ, sym))
            summed = firedrake.ffc_interface.sum_integrands(expr.form)
            kernels[expr] = []
            mapper = RemoveRestrictions()
            for i, it in enumerate(summed.integrals()):
                typ = it.integral_type()
                form = ufl.Form([it])
                prefix = "subkernel%d_%d_%s_" % (len(kernels), i, typ)
                if typ == "interior_facet":
                    newit = map_integrand_dags(mapper, it)
                    newit = newit.reconstruct(integral_type="exterior_facet")
                    form = ufl.Form([newit])

                ks = firedrake.ffc_interface.compile_form(form, prefix)
                kernels[expr].append((typ, ks))

            return
        if isinstance(expr, (UOp, BinOp, Transpose, Inverse)):
            map(get_decls, expr.operands)
            return
        raise NotImplementedError("Expression of type %s not handled",
                                  type(expr).__name__)

    get_decls(expr)

    coefficients = expr.coefficients()
    coeffmap = dict((c, ast.Symbol("w%d" % i)) for i, c in enumerate(coefficients))
    facetcoeffmap = {}
    coords = None
    coordsym = ast.Symbol("coords")
    cellfacetsym = ast.Symbol("cell_facets")
    inc = []
    for exp, sym in syms.items():
        statements.append(ast.FlatBlock("%s.setZero();\n" % sym))
        for (typ, klist) in kernels[exp]:
            for ks in klist:
                coeffs = []
                kernel = ks[-1]
                if typ not in ["cell", "interior_facet", "exterior_facet"]:
                    raise NotImplementedError("Integral type '%s' not supported" % ks[1])
                if typ in ["interior_facet", "exterior_facet"]:
                    needs_cell_facets = True
                if coords is not None:
                    assert ks[3] == coords
                else:
                    coords = ks[3]
                for coeff in ks[4]:
                    coeffs.append(coeffmap[coeff])
                inc.extend(kernel._include_dirs)
                row, col = ks[0]
                rshape = exp.shapes[0][row]
                rstart = sum(exp.shapes[0][:row])
                try:
                    cshape = exp.shapes[1][col]
                    cstart = sum(exp.shapes[1][:col])
                except:
                    cshape = 1
                    cstart = 0
                # Sub-block of mixed
                if (rshape, cshape) != exp.shape:
                    tensor = ast.FlatBlock("%s.block<%d,%d>(%d, %d)" %
                                           (sym, rshape, cshape,
                                            rstart, cstart))
                else:
                    tensor = sym
                if typ in ["exterior_facet", "interior_facet"]:
                    itsym = ast.Symbol("i0")
                    block = []
                    mesh = coords.function_space().mesh()
                    nfacet = mesh._plex.getConeSize(mesh._plex.getHeightStratum(0)[0])
                    coeffs.append(ast.FlatBlock("&%s" % itsym))
                    if typ == "exterior_facet":
                        check = 0
                    else:
                        check = 1
                    block.append(
                        ast.If(ast.Eq(ast.Symbol(cellfacetsym, rank=(itsym, )),
                                      check),
                               [ast.Block([ast.FunCall(kernel.name,
                                                       tensor, coordsym, *coeffs)],
                                          open_scope=True)])
                        )
                    loop = ast.For(ast.Decl("unsigned int", itsym, init=0),
                                   ast.Less(itsym, nfacet),
                                   ast.Incr(itsym, 1),
                                   block)
                    statements.append(loop)
                else:
                    statements.append(ast.FunCall(kernel.name,
                                                  tensor,
                                                  coordsym,
                                                  *coeffs))

    rettype = map_type(matrix_type(expr.shape))
    retsym = ast.Symbol("sym%d" % len(syms))
    retdatasym = ast.Symbol("datasym%d" % len(syms))
    result = ast.Decl(dtype, ast.Symbol(retdatasym, expr.shape))

    statements.append(ast.FlatBlock("%s %s((%s *)%s);\n" %
                                    (rettype, retsym,
                                     dtype, retdatasym)))


    def paren(val, prec=None, parent=None):
        if prec is None or parent >= prec:
            return val
        return "(%s)" % val

    def get_expr_str(expr, syms, prec=None):
        if isinstance(expr, (Scalar, Vector, Matrix)):
            return syms[expr].gencode()
        if isinstance(expr, BinOp):
            op = {operator.add: '+',
                  operator.sub: '-',
                  operator.mul: '*'}[expr.op]
            val = "%s %s %s" % (get_expr_str(expr.left, syms,
                                             prec=expr.prec),
                                op,
                                get_expr_str(expr.right, syms,
                                             prec=expr.prec))
            return paren(val, expr.prec, prec)
        if isinstance(expr, UOp):
            op = {operator.neg: '-',
                  operator.pos: '+'}[expr.op]
            val = "%s%s" % (op, get_expr_str(expr.tensor,
                                             syms,
                                             prec=expr.prec))
            return paren(val, expr.prec, prec)
        if isinstance(expr, Inverse):
            return "(%s).inverse()" % get_expr_str(expr.tensor, syms)
        if isinstance(expr, Transpose):
            return "(%s).transpose()" % get_expr_str(expr.tensor, syms)
        raise NotImplementedError

    rvalue = ast.FlatBlock(get_expr_str(expr, syms))
    statements.append(ast.Assign(retsym, rvalue))

    arglist = [result, ast.Decl("%s **" % dtype, coordsym)]
    for c in coefficients:
        typ = "%s **" % dtype
        if isinstance(c, firedrake.Constant):
            typ = "%s *" % dtype
        arglist.append(ast.Decl(typ, coeffmap[c]))

    if needs_cell_facets:
        arglist.append(ast.Decl("char *", cellfacetsym))
    kernel = ast.FunDecl("void", "foo", arglist,
                         ast.Block(statements),
                         pred=["static", "inline"])

    klist = []
    transformer = TransformKernelTensor()
    for v in kernels.values():
        for (_, ks) in v:
            for k in ks:
                kast = transformer.visit(k[-1]._ast)
                klist.append(ast.FlatBlock(kast.gencode()))

    klist.append(kernel)
    kernelast = ast.Node(klist)

    return coords, coefficients, needs_cell_facets, \
        firedrake.op2.Kernel(kernelast,
                             "foo",
                             cpp=True,
                             include_dirs=inc,
                             headers=["#include <eigen3/Eigen/Dense>"])


def assemble(expr, bcs=None):
    args = expr.arguments()
    rank = len(args)
    if rank not in (1, 2):
        raise NotImplementedError("Not implemented for rank-%d", rank)

    if any(isinstance(t.function_space(), firedrake.MixedFunctionSpace) for t in args):
        raise NotImplementedError("Mixed space assembly not implemented")

    if rank == 2:
        test, trial = expr.arguments()
        maps = tuple((test.cell_node_map(), trial.cell_node_map()))
        sparsity = firedrake.op2.Sparsity((test.function_space().dof_dset,
                                           trial.function_space().dof_dset),
                                          maps)
        tensor = firedrake.op2.Mat(sparsity, np.float64)
        tensor_arg = tensor(firedrake.op2.INC, (test.cell_node_map()[firedrake.op2.i[0]],
                                                trial.cell_node_map()[firedrake.op2.i[1]]),
                            flatten=True)
    else:
        test = args[0]
        tensor = firedrake.Function(test.function_space())
        tensor_arg = tensor.dat(firedrake.op2.INC, test.cell_node_map()[firedrake.op2.i[0]],
                                flatten=True)
    coords, coefficients, needs_cell_facets, kernel = get_kernel(expr)

    mesh = coords.function_space().mesh()

    args = [kernel, mesh.cell_set, tensor_arg,
            coords.dat(firedrake.op2.READ, coords.cell_node_map(),
                       flatten=True)]
    for c in coefficients:
        args.append(c.dat(firedrake.op2.READ, c.cell_node_map(),
                          flatten=True))

    if needs_cell_facets:
        args.append(mesh.cell_facets(firedrake.op2.READ))

    firedrake.op2.par_loop(*args)
    if bcs is not None and rank == 2:
        for bc in bcs:
            tensor.set_local_diagonal_entries(bc.nodes)
    return tensor


mesh = firedrake.UnitSquareMesh(1, 1, quadrilateral=False)
degree = 1
# RTe = firedrake.FiniteElement("RTCF", firedrake.quadrilateral, 1)
RTe = firedrake.FiniteElement("RT", firedrake.triangle, degree)
BrokenRT = firedrake.FunctionSpace(mesh, firedrake.BrokenElement(RTe))
DG = firedrake.FunctionSpace(mesh, "DG", degree - 1)
TraceRT = firedrake.FunctionSpace(mesh, firedrake.TraceElement(RTe))

W = BrokenRT * DG
sigma, u = firedrake.TrialFunctions(W)
tau,v  = firedrake.TestFunctions(W)

# u = firedrake.TrialFunction(DG)
# v = firedrake.TestFunction(DG)

lambdar = firedrake.TrialFunction(TraceRT)
gammar = firedrake.TestFunction(TraceRT)

bc = firedrake.DirichletBC(TraceRT, firedrake.Constant(0), (1, 2, 3, 4))
n = firedrake.FacetNormal(mesh)

mass = firedrake.dot(sigma, tau)*firedrake.dx + u*v*firedrake.dx

div = firedrake.div(sigma)*v*firedrake.dx
grad = firedrake.div(tau)*u*firedrake.dx

trace = firedrake.jump(tau, n=n)*lambdar('+')*firedrake.dS

positive_trace = firedrake.dot(tau, n)('+')*lambdar('+')*firedrake.dS

cell_hdiv = mass + div - grad
Cell_hdiv = firedrake.assemble(cell_hdiv, nest=False).M.values
positive_trace = firedrake.dot(tau, n)('+')*lambdar('+')*firedrake.dS
K = Matrix(positive_trace)

A = Matrix(cell_hdiv)
schur = -K.T * A.inv * K
cellwise = assemble(schur, bcs=[bc]).values
trace = firedrake.assemble(trace, nest=False).M.values

schur = np.dot(trace.T, np.dot(np.linalg.inv(Cell_hdiv), trace))

schur[bc.nodes, :] = 0
schur[:, bc.nodes] = 0

d = np.zeros_like(schur)
d[bc.nodes, bc.nodes] = 1
glob = d  - schur
print glob
print cellwise
print np.allclose(glob, cellwise)

# print assemble(X).values
# from IPython import embed; embed()

# VFSes

# mesh = firedrake.UnitSquareMesh(1, 1, quadrilateral=False)
# V = firedrake.VectorFunctionSpace(mesh, "DG", 0)
# Q = firedrake.FunctionSpace(mesh, "DG", 2)
# R = firedrake.FunctionSpace(mesh, "CG", 1)

# W = V
# n = firedrake.FacetNormal(mesh)

# u = firedrake.TrialFunction(W)
# v = firedrake.TestFunction(W)

# a = firedrake.inner(u, v)*firedrake.dx
# b = firedrake.jump(v, n=n)*firedrake.TrialFunction(R)('+')*firedrake.dS
# pos_b = firedrake.dot(v('+'), n('+'))*firedrake.TrialFunction(R)('+')*firedrake.dS
# # b = firedrake.dot(v, firedrake.Constant((1, 1)))*firedrake.TrialFunction(R)*firedrake.ds
# # A = firedrake.assemble(a, nest=False).M.values
# B = firedrake.assemble(b, nest=False).M.values

# # print A.shape, B.shape

# # print np.dot(B.T, np.dot(np.linalg.inv(A), B))

# print B

# b = Matrix(pos_b)

# print assemble(b).values
# print assemble(b.T * Matrix(a).inv * b).values

f = firedrake.Function(DG)
f.interpolate(firedrake.Expression("(1+8*pi*pi)*sin(x[0]*pi*2)*sin(x[1]*pi*2)"))

L = f*v*firedrake.dx

F = Vector(L)

# get_kernel(F)
rhs = K.T * Matrix(cell_hdiv).inv * F

# B = assemble(Matrix(a)).values

#print A
#print B
