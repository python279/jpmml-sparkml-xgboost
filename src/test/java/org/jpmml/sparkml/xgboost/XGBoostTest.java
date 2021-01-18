/*
 * Copyright (c) 2017 Villu Ruusmann
 *
 * This file is part of JPMML-SparkML
 *
 * JPMML-SparkML is free software: you can redistribute it and/or modify
 * it under the terms of the GNU Affero General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * JPMML-SparkML is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU Affero General Public License for more details.
 *
 * You should have received a copy of the GNU Affero General Public License
 * along with JPMML-SparkML.  If not, see <http://www.gnu.org/licenses/>.
 */
package org.jpmml.sparkml.xgboost;
import java.util.LinkedHashMap;
import java.util.Map;

import org.dmg.pmml.general_regression.GeneralRegressionModel;
import org.jpmml.evaluator.testing.PMMLEquivalence;
import org.jpmml.sparkml.model.HasRegressionTableOptions;
import org.jpmml.xgboost.HasXGBoostOptions;
import org.junit.Test;

public class XGBoostTest extends ConverterTest {
	@Override
	public Map<String, Object> getOptions(String name, String dataset){
		Map<String, Object> options = new LinkedHashMap<>();
		options.put(HasXGBoostOptions.OPTION_COMPACT, Boolean.TRUE);

		return options;
	}

	@Test
	public void evaluateXGBoostAudit() throws Exception {
		evaluate("XGBoost", "Audit");
	}

	@Test
	public void evaluateXGBoostAuto() throws Exception {
		evaluate("XGBoost", "Auto");
	}
}

