// Licensed to the Apache Software Foundation (ASF) under one or more
// contributor license agreements.  See the NOTICE file distributed with
// this work for additional information regarding copyright ownership.
// The ASF licenses this file to You under the Apache License, Version 2.0
// (the "License"); you may not use this file except in compliance with
// License.  You may obtain a copy of License at
//
//    http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing,
// software distributed under the License is distributed on an
// "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND,
// either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

use std::sync::Arc;

use arrow::array::{ArrayRef, Int32Array, StringArray};
use datafusion::{
    common::{Result, ScalarValue, cast::as_string_array},
    logical_expr::ColumnarValue,
};
use datafusion_ext_commons::df_execution_err;

/// Returns the position (1-based index) of the first occurrence of `substr` in `str`
/// Compatible with Spark's position function
///
/// # Arguments
/// * `args[0]` - The string to search in (str)
/// * `args[1]` - The substring to search for (substr)
/// * `args[2]` - The start position (optional, default is 1), 1-based
///
/// # Returns
/// * 1-based index of the first occurrence of substr, starting from the given start position
/// * 0 if not found
/// * null if any input is null
pub fn string_position(args: &[ColumnarValue]) -> Result<ColumnarValue> {
    let start = match args.get(2) {
        None => 1, // Default start position is 1
        Some(ColumnarValue::Scalar(ScalarValue::Int32(Some(start)))) => *start,
        Some(ColumnarValue::Scalar(scalar)) if scalar.is_null() => {
            return Ok(ColumnarValue::Scalar(ScalarValue::Int32(None)));
        }
        _ => df_execution_err!("position start position only supports literal int32")?,
    };

    // Adjust start to be 1-based
    let start = if start < 1 { 1 } else { start };

    let string_array = args[0].clone().into_array(1)?;
    let substr = match &args[1] {
        ColumnarValue::Scalar(ScalarValue::Utf8(Some(substr))) => substr,
        ColumnarValue::Scalar(ScalarValue::Utf8(None)) => {
            return Ok(ColumnarValue::Scalar(ScalarValue::Int32(None)));
        }
        _ => df_execution_err!("position substring only supports literal string")?,
    };

    if substr.is_empty() {
        // If substr is empty, return 1 (first position)
        let result_array = Arc::new(Int32Array::from_iter(
            as_string_array(&string_array)?
                .into_iter()
                .map(|s| s.map(|_| 1_i32)),
        ));
        return Ok(ColumnarValue::Array(result_array));
    }

    let result_array = Arc::new(Int32Array::from_iter(
        as_string_array(&string_array)?
            .into_iter()
            .map(|s| s.map(|str| {
                // Convert start from 1-based to 0-based
                let start_idx = (start - 1) as usize;
                if start_idx >= str.len() {
                    return 0_i32;
                }

                // Search for substring starting from start_idx
                if let Some(pos) = str[start_idx..].find(substr) {
                    // Convert back to 1-based index relative to original string
                    (pos + start_idx + 1) as i32
                } else {
                    0_i32
                }
            })),
    ));

    Ok(ColumnarValue::Array(result_array))
}

#[cfg(test)]
mod test {
    use std::sync::Arc;

    use arrow::array::{ArrayRef, Int32Array, StringArray};
    use datafusion::{
        common::{Result, ScalarValue},
        physical_plan::ColumnarValue,
    };

    use crate::spark_position::string_position;

    #[test]
    fn test_position_basic() -> Result<()> {
        let r = string_position(&vec![
            ColumnarValue::Array(Arc::new(StringArray::from_iter(vec![
                Some("Hello World"),
                Some("Hello World"),
                Some("Hello World"),
                Some("Hello World"),
                Some("Hello World"),
            ])),
            ColumnarValue::Scalar(ScalarValue::from("World")),
        ])?;
        let s = r.into_array(5)?;
        assert_eq!(
            as_string_array(&s)?
                .into_iter()
                .collect::<Vec<_>>(),
            vec![
                Some("Hello World"),
                Some("Hello World"),
                Some("Hello World"),
                Some("Hello World"),
                Some("Hello World"),
            ]
        );

        let int_array = s
            .as_any()
            .downcast_ref::<Int32Array>()
            .expect("Expected Int32Array");
        let result: Vec<_> = int_array.iter().collect();
        assert_eq!(result, vec![Some(7), Some(7), Some(7), Some(7), Some(7)]);
        Ok(())
    }

    #[test]
    fn test_position_not_found() -> Result<()> {
        let r = string_position(&vec![
            ColumnarValue::Array(Arc::new(StringArray::from_iter(vec![
                Some("Hello World"),
                Some("Hello World"),
            ])),
            ColumnarValue::Scalar(ScalarValue::from("xyz")),
        ])?;
        let s = r.into_array(2)?;

        let int_array = s
            .as_any()
            .downcast_ref::<Int32Array>()
            .expect("Expected Int32Array");
        let result: Vec<_> = int_array.iter().collect();
        assert_eq!(result, vec![Some(0), Some(0)]);
        Ok(())
    }

    #[test]
    fn test_position_with_start() -> Result<()> {
        let r = string_position(&vec![
            ColumnarValue::Array(Arc::new(StringArray::from_iter(vec![
                Some("Hello World World"),
                Some("Hello World World"),
                Some("Hello World World"),
            ])),
            ColumnarValue::Scalar(ScalarValue::from("World")),
            ColumnarValue::Scalar(ScalarValue::from(8_i32)),
        ])?;
        let s = r.into_array(3)?;

        let int_array = s
            .as_any()
            .downcast_ref::<Int32Array>()
            .expect("Expected Int32Array");
        let result: Vec<_> = int_array.iter().collect();
        assert_eq!(result, vec![Some(13), Some(13), Some(13)]);
        Ok(())
    }

    #[test]
    fn test_position_empty_substring() -> Result<()> {
        let r = string_position(&vec![
            ColumnarValue::Array(Arc::new(StringArray::from_iter(vec![
                Some("Hello"),
                Some(""),
                None,
            ])),
            ColumnarValue::Scalar(ScalarValue::from("")),
        ])?;
        let s = r.into_array(3)?;

        let int_array = s
            .as_any()
            .downcast_ref::<Int32Array>()
            .expect("Expected Int32Array");
        let result: Vec<_> = int_array.iter().collect();
        assert_eq!(result, vec![Some(1), Some(1), None]);
        Ok(())
    }

    #[test]
    fn test_position_null_input() -> Result<()> {
        // Null string array
        let r = string_position(&vec![
            ColumnarValue::Scalar(ScalarValue::Utf8(None)),
            ColumnarValue::Scalar(ScalarValue::from("World")),
        ])?;
        match r {
            ColumnarValue::Scalar(ScalarValue::Int32(None)) => Ok(()),
            other => panic!("Expected null Int32 scalar, got: {:?}", other),
        }
    }

    #[test]
    fn test_position_null_substring() -> Result<()> {
        let r = string_position(&vec![
            ColumnarValue::Array(Arc::new(StringArray::from_iter(vec![
                Some("Hello World"),
                Some("Hello World"),
            ])),
            ColumnarValue::Scalar(ScalarValue::Utf8(None)),
        ])?;
        match r {
            ColumnarValue::Scalar(ScalarValue::Int32(None)) => Ok(()),
            other => panic!("Expected null Int32 scalar, got: {:?}", other),
        }
    }

    #[test]
    fn test_position_case_sensitive() -> Result<()> {
        let r = string_position(&vec![
            ColumnarValue::Array(Arc::new(StringArray::from_iter(vec![
                Some("Hello World"),
                Some("Hello World"),
            ])),
            ColumnarValue::Scalar(ScalarValue::from("hello")),
        ])?;
        let s = r.into_array(2)?;

        let int_array = s
            .as_any()
            .downcast_ref::<Int32Array>()
            .expect("Expected Int32Array");
        let result: Vec<_> = int_array.iter().collect();
        assert_eq!(result, vec![Some(0), Some(0)]);
        Ok(())
    }

    #[test]
    fn test_position_multiple_occurrences() -> Result<()> {
        let r = string_position(&vec![
            ColumnarValue::Array(Arc::new(StringArray::from_iter(vec![
                Some("banana"),
                Some("banana"),
                Some("banana"),
            ])),
            ColumnarValue::Scalar(ScalarValue::from("na")),
        ])?;
        let s = r.into_array(3)?;

        let int_array = s
            .as_any()
            .downcast_ref::<Int32Array>()
            .expect("Expected Int32Array");
        let result: Vec<_> = int_array.iter().collect();
        assert_eq!(result, vec![Some(3), Some(3), Some(3)]);
        Ok(())
    }

    #[test]
    fn test_position_special_characters() -> Result<()> {
        let r = string_position(&vec![
            ColumnarValue::Array(Arc::new(StringArray::from_iter(vec![
                Some("a,b,c,d"),
                Some("a,b,c,d"),
                Some("a,b,c,d"),
            ])),
            ColumnarValue::Scalar(ScalarValue::from(",")),
        ])?;
        let s = r.into_array(3)?;

        let int_array = s
            .as_any()
            .downcast_ref::<Int32Array>()
            .expect("Expected Int32Array");
        let result: Vec<_> = int_array.iter().collect();
        assert_eq!(result, vec![Some(2), Some(2), Some(2)]);
        Ok(())
    }

    #[test]
    fn test_position_start_beyond_length() -> Result<()> {
        let r = string_position(&vec![
            ColumnarValue::Array(Arc::new(StringArray::from_iter(vec![
                Some("Hello"),
                Some("Hello"),
            ])),
            ColumnarValue::Scalar(ScalarValue::from("l")),
            ColumnarValue::Scalar(ScalarValue::from(100_i32)),
        ])?;
        let s = r.into_array(2)?;

        let int_array = s
            .as_any()
            .downcast_ref::<Int32Array>()
            .expect("Expected Int32Array");
        let result: Vec<_> = int_array.iter().collect();
        assert_eq!(result, vec![Some(0), Some(0)]);
        Ok(())
    }

    #[test]
    fn test_position_negative_start() -> Result<()> {
        // Negative start should be treated as 1
        let r = string_position(&vec![
            ColumnarValue::Array(Arc::new(StringArray::from_iter(vec![
                Some("Hello World"),
                Some("Hello World"),
            ])),
            ColumnarValue::Scalar(ScalarValue::from("World")),
            ColumnarValue::Scalar(ScalarValue::from(-5_i32)),
        ])?;
        let s = r.into_array(2)?;

        let int_array = s
            .as_any()
            .downcast_ref::<Int32Array>()
            .expect("Expected Int32Array");
        let result: Vec<_> = int_array.iter().collect();
        assert_eq!(result, vec![Some(7), Some(7)]);
        Ok(())
    }

    #[test]
    fn test_position_unicode() -> Result<()> {
        let r = string_position(&vec![
            ColumnarValue::Array(Arc::new(StringArray::from_iter(vec![
                Some("你好世界"),
                Some("你好世界"),
                Some("你好世界"),
            ])),
            ColumnarValue::Scalar(ScalarValue::from("世界")),
        ])?;
        let s = r.into_array(3)?;

        let int_array = s
            .as_any()
            .downcast_ref::<Int32Array>()
            .expect("Expected Int32Array");
        let result: Vec<_> = int_array.iter().collect();
        assert_eq!(result, vec![Some(3), Some(3), Some(3)]);
        Ok(())
    }

    #[test]
    fn test_position_chinese_mixed() -> Result<()> {
        let r = string_position(&vec![
            ColumnarValue::Array(Arc::new(StringArray::from_iter(vec![
                Some("张三Hello李四"),
                Some("张三Hello李四"),
            ])),
            ColumnarValue::Scalar(ScalarValue::from("Hello")),
        ])?;
        let s = r.into_array(2)?;

        let int_array = s
            .as_any()
            .downcast_ref::<Int32Array>()
            .expect("Expected Int32Array");
        let result: Vec<_> = int_array.iter().collect();
        assert_eq!(result, vec![Some(3), Some(3)]);
        Ok(())
    }
}
