use super::prelude::*;
use crate::burn::TensorKind;
use onnx_ir::scatter_elements::ScatterElementsReduction;

impl NodeCodegen for onnx_ir::scatter_elements::ScatterElementsNode {
    fn inputs(&self) -> &[Argument] {
        &self.inputs
    }

    fn outputs(&self) -> &[Argument] {
        &self.outputs
    }

    fn forward(&self, scope: &mut ScopeAtPosition<'_>) -> TokenStream {
        let axis = self.config.axis.to_tokens();
        let data = scope.arg(self.inputs.first().unwrap());
        let indices = scope.arg(&self.inputs[1]);
        let updates = scope.arg(&self.inputs[2]);
        let output = arg_to_ident(self.outputs.first().unwrap());

        let data_arg = self.inputs.first().unwrap();
        let (data_kind, rank) = match &data_arg.ty {
            ArgType::Tensor(t) => (TensorKind::from(t.dtype), t.rank),
            _ => panic!("Expected tensor input for data"),
        };
        let rank_lit = rank.to_tokens();

        if matches!(data_kind, TensorKind::Bool)
            && !matches!(self.config.reduction, ScatterElementsReduction::None)
        {
            panic!(
                "ScatterElements with {:?} reduction not supported for bool tensors",
                self.config.reduction
            );
        }

        // ONNX allows indices down to `-dim_size` along the scatter axis, which burn's
        // indexing does not accept, so fold negatives before scattering.
        let prologue = quote! {
            let se_data = #data;
            let se_axis_size = se_data.dims()[#axis] as i64;
            let se_indices = #indices.cast(burn::tensor::DType::I64);
            let se_negative = se_indices.clone().lower_elem(0i64);
            let se_corrected = se_indices.clone() + se_axis_size;
            let se_indices = se_indices.mask_where(se_negative, se_corrected);
        };

        // `Add` is the one update op every backend implements for element-wise `scatter`.
        if matches!(self.config.reduction, ScatterElementsReduction::Add) {
            return quote! {
                let #output = {
                    #prologue
                    se_data.scatter(#axis, se_indices, #updates, burn::tensor::IndexingUpdateOp::Add)
                };
            };
        }

        let update_op = match &self.config.reduction {
            ScatterElementsReduction::None => quote! { burn::tensor::IndexingUpdateOp::Assign },
            ScatterElementsReduction::Mul => quote! { burn::tensor::IndexingUpdateOp::Mul },
            ScatterElementsReduction::Max => quote! { burn::tensor::IndexingUpdateOp::Max },
            ScatterElementsReduction::Min => quote! { burn::tensor::IndexingUpdateOp::Min },
            ScatterElementsReduction::Add => unreachable!("handled above"),
        };

        // Assign, Mul, Min and Max are unimplemented for element-wise `scatter` in the flex
        // and cubecl backends (tracel-ai/burn#5522), but `scatter_nd` implements all of them
        // everywhere. ScatterElements writes to
        //   data[p_0, .., indices[p], .., p_{r-1}] for every p in the index shape,
        // so materializing those coordinates as index tuples turns it into a ScatterND. The
        // non-axis columns are the row-major coordinates of p, recovered from a flat arange.
        let coordinates = quote! {
            let se_idx_dims = se_indices.dims();
            let se_n: usize = se_idx_dims.iter().product();
            let mut se_strides = [1usize; #rank_lit];
            for se_d in (0..#rank_lit - 1).rev() {
                se_strides[se_d] = se_strides[se_d + 1] * se_idx_dims[se_d + 1];
            }
            let se_flat = Tensor::<1, Int>::arange(
                0..se_n as i64,
                (&self.device, burn::tensor::DType::I64),
            );
            let mut se_columns: alloc::vec::Vec<Tensor<2, Int>> =
                alloc::vec::Vec::with_capacity(#rank_lit);
            for se_d in 0..#rank_lit {
                se_columns.push(if se_d == #axis {
                    se_indices.clone().reshape([se_n, 1])
                } else {
                    se_flat
                        .clone()
                        .div_scalar(se_strides[se_d] as i64)
                        .remainder_scalar(se_idx_dims[se_d] as i64)
                        .reshape([se_n, 1])
                });
            }
            let se_coordinates = Tensor::cat(se_columns, 1);
        };

        if matches!(data_kind, TensorKind::Bool) {
            // `scatter_nd` panics for bool tensors, so round-trip through i64.
            return quote! {
                let #output = {
                    #prologue
                    #coordinates
                    let se_updates = #updates.int().cast(burn::tensor::DType::I64);
                    se_data
                        .int()
                        .cast(burn::tensor::DType::I64)
                        .scatter_nd(se_coordinates, se_updates.reshape([se_n]), #update_op)
                        .bool()
                };
            };
        }

        quote! {
            let #output = {
                #prologue
                #coordinates
                se_data.scatter_nd(se_coordinates, #updates.reshape([se_n]), #update_op)
            };
        }
    }
}

#[cfg(test)]
mod tests {
    use super::super::test_helpers::*;
    use burn::tensor::{BoolStore, DType};
    use insta::assert_snapshot;
    use onnx_ir::scatter_elements::{
        ScatterElementsConfig, ScatterElementsNodeBuilder, ScatterElementsReduction,
    };

    #[test]
    fn test_scatter_elements_none() {
        let config = ScatterElementsConfig::new(0, ScatterElementsReduction::None);
        let node = ScatterElementsNodeBuilder::new("scatter1")
            .input_tensor("data", 2, DType::F32)
            .input_tensor("indices", 2, DType::I64)
            .input_tensor("updates", 2, DType::F32)
            .output_tensor("output", 2, DType::F32)
            .config(config)
            .build();
        let code = codegen_forward_default(&node);
        assert_snapshot!(code, @r"
        pub fn forward(
            &self,
            data: Tensor<2>,
            indices: Tensor<2, Int>,
            updates: Tensor<2>,
        ) -> Tensor<2> {
            let output = {
                let se_data = data;
                let se_axis_size = se_data.dims()[0] as i64;
                let se_indices = indices.cast(burn::tensor::DType::I64);
                let se_negative = se_indices.clone().lower_elem(0i64);
                let se_corrected = se_indices.clone() + se_axis_size;
                let se_indices = se_indices.mask_where(se_negative, se_corrected);
                let se_idx_dims = se_indices.dims();
                let se_n: usize = se_idx_dims.iter().product();
                let mut se_strides = [1usize; 2];
                for se_d in (0..2 - 1).rev() {
                    se_strides[se_d] = se_strides[se_d + 1] * se_idx_dims[se_d + 1];
                }
                let se_flat = Tensor::<
                    1,
                    Int,
                >::arange(0..se_n as i64, (&self.device, burn::tensor::DType::I64));
                let mut se_columns: alloc::vec::Vec<Tensor<2, Int>> = alloc::vec::Vec::with_capacity(
                    2,
                );
                for se_d in 0..2 {
                    se_columns
                        .push(
                            if se_d == 0 {
                                se_indices.clone().reshape([se_n, 1])
                            } else {
                                se_flat
                                    .clone()
                                    .div_scalar(se_strides[se_d] as i64)
                                    .remainder_scalar(se_idx_dims[se_d] as i64)
                                    .reshape([se_n, 1])
                            },
                        );
                }
                let se_coordinates = Tensor::cat(se_columns, 1);
                se_data
                    .scatter_nd(
                        se_coordinates,
                        updates.reshape([se_n]),
                        burn::tensor::IndexingUpdateOp::Assign,
                    )
            };
            output
        }
        ");
    }

    #[test]
    fn test_scatter_elements_add() {
        let config = ScatterElementsConfig::new(1, ScatterElementsReduction::Add);
        let node = ScatterElementsNodeBuilder::new("scatter1")
            .input_tensor("data", 2, DType::F32)
            .input_tensor("indices", 2, DType::I64)
            .input_tensor("updates", 2, DType::F32)
            .output_tensor("output", 2, DType::F32)
            .config(config)
            .build();
        let code = codegen_forward_default(&node);
        assert_snapshot!(code, @r"
        pub fn forward(
            &self,
            data: Tensor<2>,
            indices: Tensor<2, Int>,
            updates: Tensor<2>,
        ) -> Tensor<2> {
            let output = {
                let se_data = data;
                let se_axis_size = se_data.dims()[1] as i64;
                let se_indices = indices.cast(burn::tensor::DType::I64);
                let se_negative = se_indices.clone().lower_elem(0i64);
                let se_corrected = se_indices.clone() + se_axis_size;
                let se_indices = se_indices.mask_where(se_negative, se_corrected);
                se_data.scatter(1, se_indices, updates, burn::tensor::IndexingUpdateOp::Add)
            };
            output
        }
        ");
    }

    #[test]
    fn test_scatter_elements_mul() {
        let config = ScatterElementsConfig::new(0, ScatterElementsReduction::Mul);
        let node = ScatterElementsNodeBuilder::new("scatter1")
            .input_tensor("data", 2, DType::F32)
            .input_tensor("indices", 2, DType::I64)
            .input_tensor("updates", 2, DType::F32)
            .output_tensor("output", 2, DType::F32)
            .config(config)
            .build();
        let code = codegen_forward_default(&node);
        assert_snapshot!(code, @r"
        pub fn forward(
            &self,
            data: Tensor<2>,
            indices: Tensor<2, Int>,
            updates: Tensor<2>,
        ) -> Tensor<2> {
            let output = {
                let se_data = data;
                let se_axis_size = se_data.dims()[0] as i64;
                let se_indices = indices.cast(burn::tensor::DType::I64);
                let se_negative = se_indices.clone().lower_elem(0i64);
                let se_corrected = se_indices.clone() + se_axis_size;
                let se_indices = se_indices.mask_where(se_negative, se_corrected);
                let se_idx_dims = se_indices.dims();
                let se_n: usize = se_idx_dims.iter().product();
                let mut se_strides = [1usize; 2];
                for se_d in (0..2 - 1).rev() {
                    se_strides[se_d] = se_strides[se_d + 1] * se_idx_dims[se_d + 1];
                }
                let se_flat = Tensor::<
                    1,
                    Int,
                >::arange(0..se_n as i64, (&self.device, burn::tensor::DType::I64));
                let mut se_columns: alloc::vec::Vec<Tensor<2, Int>> = alloc::vec::Vec::with_capacity(
                    2,
                );
                for se_d in 0..2 {
                    se_columns
                        .push(
                            if se_d == 0 {
                                se_indices.clone().reshape([se_n, 1])
                            } else {
                                se_flat
                                    .clone()
                                    .div_scalar(se_strides[se_d] as i64)
                                    .remainder_scalar(se_idx_dims[se_d] as i64)
                                    .reshape([se_n, 1])
                            },
                        );
                }
                let se_coordinates = Tensor::cat(se_columns, 1);
                se_data
                    .scatter_nd(
                        se_coordinates,
                        updates.reshape([se_n]),
                        burn::tensor::IndexingUpdateOp::Mul,
                    )
            };
            output
        }
        ");
    }

    #[test]
    fn test_scatter_elements_max() {
        let config = ScatterElementsConfig::new(0, ScatterElementsReduction::Max);
        let node = ScatterElementsNodeBuilder::new("scatter1")
            .input_tensor("data", 2, DType::F32)
            .input_tensor("indices", 2, DType::I64)
            .input_tensor("updates", 2, DType::F32)
            .output_tensor("output", 2, DType::F32)
            .config(config)
            .build();
        let code = codegen_forward_default(&node);
        assert_snapshot!(code, @r"
        pub fn forward(
            &self,
            data: Tensor<2>,
            indices: Tensor<2, Int>,
            updates: Tensor<2>,
        ) -> Tensor<2> {
            let output = {
                let se_data = data;
                let se_axis_size = se_data.dims()[0] as i64;
                let se_indices = indices.cast(burn::tensor::DType::I64);
                let se_negative = se_indices.clone().lower_elem(0i64);
                let se_corrected = se_indices.clone() + se_axis_size;
                let se_indices = se_indices.mask_where(se_negative, se_corrected);
                let se_idx_dims = se_indices.dims();
                let se_n: usize = se_idx_dims.iter().product();
                let mut se_strides = [1usize; 2];
                for se_d in (0..2 - 1).rev() {
                    se_strides[se_d] = se_strides[se_d + 1] * se_idx_dims[se_d + 1];
                }
                let se_flat = Tensor::<
                    1,
                    Int,
                >::arange(0..se_n as i64, (&self.device, burn::tensor::DType::I64));
                let mut se_columns: alloc::vec::Vec<Tensor<2, Int>> = alloc::vec::Vec::with_capacity(
                    2,
                );
                for se_d in 0..2 {
                    se_columns
                        .push(
                            if se_d == 0 {
                                se_indices.clone().reshape([se_n, 1])
                            } else {
                                se_flat
                                    .clone()
                                    .div_scalar(se_strides[se_d] as i64)
                                    .remainder_scalar(se_idx_dims[se_d] as i64)
                                    .reshape([se_n, 1])
                            },
                        );
                }
                let se_coordinates = Tensor::cat(se_columns, 1);
                se_data
                    .scatter_nd(
                        se_coordinates,
                        updates.reshape([se_n]),
                        burn::tensor::IndexingUpdateOp::Max,
                    )
            };
            output
        }
        ");
    }

    #[test]
    fn test_scatter_elements_min() {
        let config = ScatterElementsConfig::new(0, ScatterElementsReduction::Min);
        let node = ScatterElementsNodeBuilder::new("scatter1")
            .input_tensor("data", 2, DType::F32)
            .input_tensor("indices", 2, DType::I64)
            .input_tensor("updates", 2, DType::F32)
            .output_tensor("output", 2, DType::F32)
            .config(config)
            .build();
        let code = codegen_forward_default(&node);
        assert_snapshot!(code, @r"
        pub fn forward(
            &self,
            data: Tensor<2>,
            indices: Tensor<2, Int>,
            updates: Tensor<2>,
        ) -> Tensor<2> {
            let output = {
                let se_data = data;
                let se_axis_size = se_data.dims()[0] as i64;
                let se_indices = indices.cast(burn::tensor::DType::I64);
                let se_negative = se_indices.clone().lower_elem(0i64);
                let se_corrected = se_indices.clone() + se_axis_size;
                let se_indices = se_indices.mask_where(se_negative, se_corrected);
                let se_idx_dims = se_indices.dims();
                let se_n: usize = se_idx_dims.iter().product();
                let mut se_strides = [1usize; 2];
                for se_d in (0..2 - 1).rev() {
                    se_strides[se_d] = se_strides[se_d + 1] * se_idx_dims[se_d + 1];
                }
                let se_flat = Tensor::<
                    1,
                    Int,
                >::arange(0..se_n as i64, (&self.device, burn::tensor::DType::I64));
                let mut se_columns: alloc::vec::Vec<Tensor<2, Int>> = alloc::vec::Vec::with_capacity(
                    2,
                );
                for se_d in 0..2 {
                    se_columns
                        .push(
                            if se_d == 0 {
                                se_indices.clone().reshape([se_n, 1])
                            } else {
                                se_flat
                                    .clone()
                                    .div_scalar(se_strides[se_d] as i64)
                                    .remainder_scalar(se_idx_dims[se_d] as i64)
                                    .reshape([se_n, 1])
                            },
                        );
                }
                let se_coordinates = Tensor::cat(se_columns, 1);
                se_data
                    .scatter_nd(
                        se_coordinates,
                        updates.reshape([se_n]),
                        burn::tensor::IndexingUpdateOp::Min,
                    )
            };
            output
        }
        ");
    }

    #[test]
    fn test_scatter_elements_int() {
        let config = ScatterElementsConfig::new(0, ScatterElementsReduction::None);
        let node = ScatterElementsNodeBuilder::new("scatter1")
            .input_tensor("data", 2, DType::I64)
            .input_tensor("indices", 2, DType::I64)
            .input_tensor("updates", 2, DType::I64)
            .output_tensor("output", 2, DType::I64)
            .config(config)
            .build();
        let code = codegen_forward_default(&node);
        assert_snapshot!(code, @r"
        pub fn forward(
            &self,
            data: Tensor<2, Int>,
            indices: Tensor<2, Int>,
            updates: Tensor<2, Int>,
        ) -> Tensor<2, Int> {
            let output = {
                let se_data = data;
                let se_axis_size = se_data.dims()[0] as i64;
                let se_indices = indices.cast(burn::tensor::DType::I64);
                let se_negative = se_indices.clone().lower_elem(0i64);
                let se_corrected = se_indices.clone() + se_axis_size;
                let se_indices = se_indices.mask_where(se_negative, se_corrected);
                let se_idx_dims = se_indices.dims();
                let se_n: usize = se_idx_dims.iter().product();
                let mut se_strides = [1usize; 2];
                for se_d in (0..2 - 1).rev() {
                    se_strides[se_d] = se_strides[se_d + 1] * se_idx_dims[se_d + 1];
                }
                let se_flat = Tensor::<
                    1,
                    Int,
                >::arange(0..se_n as i64, (&self.device, burn::tensor::DType::I64));
                let mut se_columns: alloc::vec::Vec<Tensor<2, Int>> = alloc::vec::Vec::with_capacity(
                    2,
                );
                for se_d in 0..2 {
                    se_columns
                        .push(
                            if se_d == 0 {
                                se_indices.clone().reshape([se_n, 1])
                            } else {
                                se_flat
                                    .clone()
                                    .div_scalar(se_strides[se_d] as i64)
                                    .remainder_scalar(se_idx_dims[se_d] as i64)
                                    .reshape([se_n, 1])
                            },
                        );
                }
                let se_coordinates = Tensor::cat(se_columns, 1);
                se_data
                    .scatter_nd(
                        se_coordinates,
                        updates.reshape([se_n]),
                        burn::tensor::IndexingUpdateOp::Assign,
                    )
            };
            output
        }
        ");
    }

    #[test]
    fn test_scatter_elements_bool_none() {
        let config = ScatterElementsConfig::new(0, ScatterElementsReduction::None);
        let node = ScatterElementsNodeBuilder::new("scatter1")
            .input_tensor("data", 1, DType::Bool(BoolStore::Native))
            .input_tensor("indices", 1, DType::I64)
            .input_tensor("updates", 1, DType::Bool(BoolStore::Native))
            .output_tensor("output", 1, DType::Bool(BoolStore::Native))
            .config(config)
            .build();
        let code = codegen_forward_default(&node);
        assert_snapshot!(code, @r"
        pub fn forward(
            &self,
            data: Tensor<1, Bool>,
            indices: Tensor<1, Int>,
            updates: Tensor<1, Bool>,
        ) -> Tensor<1, Bool> {
            let output = {
                let se_data = data;
                let se_axis_size = se_data.dims()[0] as i64;
                let se_indices = indices.cast(burn::tensor::DType::I64);
                let se_negative = se_indices.clone().lower_elem(0i64);
                let se_corrected = se_indices.clone() + se_axis_size;
                let se_indices = se_indices.mask_where(se_negative, se_corrected);
                let se_idx_dims = se_indices.dims();
                let se_n: usize = se_idx_dims.iter().product();
                let mut se_strides = [1usize; 1];
                for se_d in (0..1 - 1).rev() {
                    se_strides[se_d] = se_strides[se_d + 1] * se_idx_dims[se_d + 1];
                }
                let se_flat = Tensor::<
                    1,
                    Int,
                >::arange(0..se_n as i64, (&self.device, burn::tensor::DType::I64));
                let mut se_columns: alloc::vec::Vec<Tensor<2, Int>> = alloc::vec::Vec::with_capacity(
                    1,
                );
                for se_d in 0..1 {
                    se_columns
                        .push(
                            if se_d == 0 {
                                se_indices.clone().reshape([se_n, 1])
                            } else {
                                se_flat
                                    .clone()
                                    .div_scalar(se_strides[se_d] as i64)
                                    .remainder_scalar(se_idx_dims[se_d] as i64)
                                    .reshape([se_n, 1])
                            },
                        );
                }
                let se_coordinates = Tensor::cat(se_columns, 1);
                let se_updates = updates.int().cast(burn::tensor::DType::I64);
                se_data
                    .int()
                    .cast(burn::tensor::DType::I64)
                    .scatter_nd(
                        se_coordinates,
                        se_updates.reshape([se_n]),
                        burn::tensor::IndexingUpdateOp::Assign,
                    )
                    .bool()
            };
            output
        }
        ");
    }

    #[test]
    #[should_panic(expected = "reduction not supported for bool tensors")]
    fn test_scatter_elements_bool_add_panics() {
        let config = ScatterElementsConfig::new(0, ScatterElementsReduction::Add);
        let node = ScatterElementsNodeBuilder::new("scatter1")
            .input_tensor("data", 1, DType::Bool(BoolStore::Native))
            .input_tensor("indices", 1, DType::I64)
            .input_tensor("updates", 1, DType::Bool(BoolStore::Native))
            .output_tensor("output", 1, DType::Bool(BoolStore::Native))
            .config(config)
            .build();
        codegen_forward_default(&node);
    }
}
