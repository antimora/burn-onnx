//! Guards generated code against a temporary shadowing a graph value.
//!
//! Graph inputs keep their sanitized ONNX names as `forward()` parameters, so
//! a temporary that node codegen declares in a block can shadow a graph value
//! the same scope reads afterwards: `let k = ...; input.topk(k)` is wrong when
//! the data input itself is named `k`. Nothing in the emitted tokens tells a
//! temporary apart from a graph value, so every graph value reference is
//! emitted with a tag ([`value_ident`]), the assembled body is walked with a
//! scope stack ([`Checker`]), and the tag is removed before the code is
//! written out ([`strip`]).
//!
//! `sanitize_name` collapses consecutive underscores, so no graph value can
//! carry the tag, and codegen temporaries never use a `__` prefix.

use core::fmt;
use proc_macro2::{Group, Ident, Span, TokenStream, TokenTree};
use quote::quote;
use syn::visit::Visit;

const TAG: &str = "__arg_";

/// The identifier that generated code uses to refer to the graph value `name`.
pub(crate) fn value_ident(name: &str) -> Ident {
    Ident::new(&format!("{TAG}{name}"), Span::call_site())
}

/// The graph value name behind a tagged identifier, if it is one.
fn tagged(ident: &Ident) -> Option<String> {
    ident.to_string().strip_prefix(TAG).map(str::to_string)
}

/// Replace every tagged identifier with the plain graph value name.
pub(crate) fn strip(tokens: TokenStream) -> TokenStream {
    tokens
        .into_iter()
        .map(|tree| match tree {
            TokenTree::Ident(ident) => match tagged(&ident) {
                Some(name) => TokenTree::Ident(Ident::new(&name, ident.span())),
                None => TokenTree::Ident(ident),
            },
            TokenTree::Group(group) => {
                let mut stripped = Group::new(group.delimiter(), strip(group.stream()));
                stripped.set_span(group.span());
                TokenTree::Group(stripped)
            }
            other => other,
        })
        .collect()
}

/// A graph value read while a temporary of the same name is in scope.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct Shadowed {
    /// Where the read happens, e.g. "node `topk1`" or "the forward() return".
    pub site: String,
    pub name: String,
}

impl fmt::Display for Shadowed {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "generated code for {} reads the graph value `{}` while a local named `{}` is \
             in scope. Rename the temporary in the node's codegen, or rename the ONNX value.",
            self.site, self.name, self.name
        )
    }
}

/// What a name in scope currently resolves to.
///
/// A tagged binding (`let __arg_x = ...`) is a graph value; a plain one is a
/// temporary. Later bindings shadow earlier ones, so a graph value bound after
/// a same-named temporary makes the name safe to read again.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Binding {
    Value,
    Temporary,
}

/// Walks each slice of a forward body in order, carrying the bindings made at
/// `forward()` scope from one slice to the next.
#[derive(Debug, Default)]
pub(crate) struct Checker {
    function_scope: Vec<(String, Binding)>,
}

impl Checker {
    /// Check the statements of one site (a node, or the trailing output code),
    /// which follow every site checked before.
    ///
    /// Tokens that do not parse as statements are skipped: they will fail to
    /// compile with a better message than this check could give.
    pub(crate) fn check(&mut self, site: &str, body: &TokenStream) -> Result<(), Shadowed> {
        let block: syn::Block = match syn::parse2(quote! { { #body } }) {
            Ok(block) => block,
            Err(err) => {
                log::debug!("Skipping shadow check for {site}: {err}");
                return Ok(());
            }
        };
        let mut walk = Walk {
            site,
            function_scope: &mut self.function_scope,
            scopes: Vec::new(),
            shadowed: None,
        };
        // The outer block is the node's slice of forward(), not a scope of its own.
        syn::visit::visit_block(&mut walk, &block);
        match walk.shadowed {
            Some(shadowed) => Err(shadowed),
            None => Ok(()),
        }
    }
}

struct Walk<'a> {
    site: &'a str,
    function_scope: &'a mut Vec<(String, Binding)>,
    scopes: Vec<Vec<(String, Binding)>>,
    shadowed: Option<Shadowed>,
}

impl Walk<'_> {
    fn declare(&mut self, name: String, binding: Binding) {
        match self.scopes.last_mut() {
            Some(scope) => scope.push((name, binding)),
            None => self.function_scope.push((name, binding)),
        }
    }

    /// The innermost, latest binding of `name`, as Rust would resolve it.
    fn resolve(&self, name: &str) -> Option<Binding> {
        self.scopes
            .iter()
            .rev()
            .chain(core::iter::once(&*self.function_scope))
            .find_map(|scope| {
                scope
                    .iter()
                    .rev()
                    .find(|(bound, _)| bound == name)
                    .map(|(_, binding)| *binding)
            })
    }

    fn read(&mut self, ident: &Ident) {
        if self.shadowed.is_some() {
            return;
        }
        if let Some(name) = tagged(ident)
            && self.resolve(&name) == Some(Binding::Temporary)
        {
            self.shadowed = Some(Shadowed {
                site: self.site.to_string(),
                name,
            });
        }
    }

    /// Declare every identifier the pattern binds.
    fn bind(&mut self, pat: &syn::Pat) {
        let mut names = PatNames::default();
        names.visit_pat(pat);
        for (name, binding) in names.0 {
            self.declare(name, binding);
        }
    }

    fn scoped(&mut self, f: impl FnOnce(&mut Self)) {
        self.scopes.push(Vec::new());
        f(self);
        self.scopes.pop();
    }

    /// Macro bodies are opaque to syn; scan them for tagged reads.
    fn read_tokens(&mut self, tokens: TokenStream) {
        for tree in tokens {
            match tree {
                TokenTree::Ident(ident) => self.read(&ident),
                TokenTree::Group(group) => self.read_tokens(group.stream()),
                _ => {}
            }
        }
    }
}

impl<'ast> Visit<'ast> for Walk<'_> {
    fn visit_block(&mut self, block: &'ast syn::Block) {
        self.scoped(|walk| syn::visit::visit_block(walk, block));
    }

    fn visit_local(&mut self, local: &'ast syn::Local) {
        // The initializer runs before the binding exists.
        if let Some(init) = &local.init {
            self.visit_expr(&init.expr);
            if let Some((_, diverge)) = &init.diverge {
                self.visit_expr(diverge);
            }
        }
        self.bind(&local.pat);
    }

    fn visit_expr_let(&mut self, expr: &'ast syn::ExprLet) {
        self.visit_expr(&expr.expr);
        self.bind(&expr.pat);
    }

    fn visit_expr_if(&mut self, expr: &'ast syn::ExprIf) {
        // An `if let` binding is visible in the then branch only.
        self.scoped(|walk| {
            walk.visit_expr(&expr.cond);
            walk.visit_block(&expr.then_branch);
        });
        if let Some((_, else_branch)) = &expr.else_branch {
            self.visit_expr(else_branch);
        }
    }

    fn visit_expr_while(&mut self, expr: &'ast syn::ExprWhile) {
        self.scoped(|walk| {
            walk.visit_expr(&expr.cond);
            walk.visit_block(&expr.body);
        });
    }

    fn visit_expr_for_loop(&mut self, expr: &'ast syn::ExprForLoop) {
        self.visit_expr(&expr.expr);
        self.scoped(|walk| {
            walk.bind(&expr.pat);
            walk.visit_block(&expr.body);
        });
    }

    fn visit_expr_closure(&mut self, expr: &'ast syn::ExprClosure) {
        self.scoped(|walk| {
            for input in &expr.inputs {
                walk.bind(input);
            }
            walk.visit_expr(&expr.body);
        });
    }

    fn visit_arm(&mut self, arm: &'ast syn::Arm) {
        self.scoped(|walk| {
            walk.bind(&arm.pat);
            // Visiting the pattern reaches the reads inside a guard.
            walk.visit_pat(&arm.pat);
            walk.visit_expr(&arm.body);
        });
    }

    fn visit_expr_path(&mut self, path: &'ast syn::ExprPath) {
        if path.qself.is_none()
            && path.path.leading_colon.is_none()
            && path.path.segments.len() == 1
        {
            self.read(&path.path.segments[0].ident);
        }
    }

    fn visit_macro(&mut self, mac: &'ast syn::Macro) {
        self.read_tokens(mac.tokens.clone());
    }
}

/// Collects the identifiers a pattern binds, tagged ones as graph values.
#[derive(Default)]
struct PatNames(Vec<(String, Binding)>);

impl<'ast> Visit<'ast> for PatNames {
    fn visit_pat_ident(&mut self, pat: &'ast syn::PatIdent) {
        self.0.push(match tagged(&pat.ident) {
            Some(name) => (name, Binding::Value),
            None => (pat.ident.to_string(), Binding::Temporary),
        });
        syn::visit::visit_pat_ident(self, pat);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn arg(name: &str) -> Ident {
        value_ident(name)
    }

    fn check(body: TokenStream) -> Result<(), Shadowed> {
        Checker::default().check("node1", &body)
    }

    fn shadowed(name: &str) -> Result<(), Shadowed> {
        Err(Shadowed {
            site: "node1".to_string(),
            name: name.to_string(),
        })
    }

    #[test]
    fn temporary_declared_before_graph_value_read() {
        let k = arg("k");
        let x = arg("x");
        let body = quote! {
            let out = {
                let k: usize = 3;
                #x.topk(k)
            };
        };
        assert_eq!(check(body), Ok(()));

        let body = quote! {
            let out = {
                let k: usize = 3;
                #k.topk(k)
            };
        };
        assert_eq!(check(body), shadowed("k"));
    }

    #[test]
    fn graph_value_rebound_into_temporary_of_same_name() {
        let indices = arg("indices");
        let body = quote! {
            let out = {
                let indices = #indices.cast(I64);
                let negative = indices.clone().lower_elem(0i64);
                indices.mask_where(negative)
            };
        };
        assert_eq!(check(body), Ok(()));
    }

    #[test]
    fn sequential_capture_of_swapped_inputs() {
        let lhs = arg("lhs");
        let rhs = arg("rhs");
        let body = quote! {
            let out = {
                let lhs = #rhs;
                let rhs = #lhs;
                lhs.add(rhs)
            };
        };
        assert_eq!(check(body), shadowed("lhs"));

        let body = quote! {
            let out = {
                let (lhs, rhs) = (#rhs, #lhs);
                lhs.add(rhs)
            };
        };
        assert_eq!(check(body), Ok(()));
    }

    #[test]
    fn block_scope_ends_with_the_block() {
        let dims = arg("dims");
        let body = quote! {
            let out1 = {
                let dims = [1usize, 2usize];
                dims[0]
            };
            let out2 = #dims.reshape([1]);
        };
        assert_eq!(check(body), Ok(()));
    }

    #[test]
    fn function_scope_temporary_reaches_later_nodes() {
        let actual_idx = arg("actual_idx");
        let mut checker = Checker::default();
        let first = quote! {
            let actual_idx = if 1 < 0 { 0usize } else { 1usize };
            let out1 = shape[actual_idx];
        };
        assert_eq!(checker.check("gather1", &first), Ok(()));
        let second = quote! {
            let out2 = #actual_idx.abs();
        };
        assert_eq!(
            checker.check("abs1", &second),
            Err(Shadowed {
                site: "abs1".to_string(),
                name: "actual_idx".to_string(),
            })
        );
    }

    #[test]
    fn graph_value_bound_after_a_temporary_takes_the_name_back() {
        let actual_idx = arg("actual_idx");
        let mut checker = Checker::default();
        let first = quote! {
            let actual_idx = 1usize;
            let out1 = shape[actual_idx];
        };
        assert_eq!(checker.check("gather1", &first), Ok(()));
        let second = quote! {
            let #actual_idx = out1.abs();
            let out2 = #actual_idx.abs();
        };
        assert_eq!(checker.check("abs1", &second), Ok(()));
    }

    #[test]
    fn reads_inside_match_guards_are_checked() {
        let x = arg("x");
        let body = quote! {
            let out = {
                let x = 1i64;
                match Some(2i64) {
                    Some(v) if v > #x => v,
                    _ => x,
                }
            };
        };
        assert_eq!(check(body), shadowed("x"));
    }

    #[test]
    fn loop_closure_and_arm_bindings_are_scoped_to_their_body() {
        let i = arg("i");
        let v = arg("v");
        let t = arg("t");
        let body = quote! {
            let out = {
                for i in 0..3usize {
                    let _ = #i;
                }
                let mapped = [1i64].map(|v| v + #v);
                let picked = match Some(1i64) {
                    Some(t) if t > #t => t,
                    _ => 0i64,
                };
                (#i, #v, #t)
            };
        };
        assert_eq!(check(body.clone()), shadowed("i"));

        let body = quote! {
            let out = {
                for i in 0..3usize {
                    let _ = i;
                }
                let mapped = [1i64].map(|v| v + 1i64);
                let picked = match Some(1i64) {
                    Some(t) if t > 0i64 => t,
                    _ => 0i64,
                };
                (#i, #v, #t)
            };
        };
        assert_eq!(check(body), Ok(()));
    }

    #[test]
    fn if_let_binding_is_scoped_to_then_branch() {
        let x = arg("x");
        let body = quote! {
            let out = if let Some(x) = Some(1i64) { x } else { #x };
            let after = #x;
        };
        assert_eq!(check(body), Ok(()));

        let body = quote! {
            let out = if let Some(x) = Some(1i64) { #x } else { 0i64 };
        };
        assert_eq!(check(body), shadowed("x"));
    }

    #[test]
    fn graph_value_rebinding_is_not_a_temporary() {
        let cond = arg("cond");
        let body = quote! {
            let out = if true {
                let #cond = #cond;
                #cond
            } else {
                #cond
            };
        };
        assert_eq!(check(body), Ok(()));
    }

    #[test]
    fn reads_inside_macros_are_checked() {
        let delta = arg("delta");
        let body = quote! {
            let out = {
                let delta = 1i64;
                assert!(#delta != 0);
                delta
            };
        };
        assert_eq!(check(body), shadowed("delta"));
    }

    #[test]
    fn strip_removes_the_tag_everywhere() {
        let x = arg("x");
        let out = arg("out");
        let tokens = quote! {
            let #out = { alloc::vec![#x.clone(), (#x)] };
        };
        assert_eq!(
            strip(tokens).to_string(),
            quote! { let out = { alloc::vec![x.clone(), (x)] }; }.to_string()
        );
    }
}
